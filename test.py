import argparse
import time
import shutil
import numpy as np

import seaborn as sns

from os import path, mkdir, remove
from functools import partial

from collections import OrderedDict

import torch
import torch.utils.data as data
import multiprocessing as mp

import tensorboardX as tensorboard

from net.dataset import CODataset, BatchSampler, collate_fn, Transform
from net.LSTM import LSTM_RAE

from utils.meters import AverageMeter
import utils.logging as logging
from utils.snapshot import resume_from_snapshot

import matplotlib.pyplot as plt


NETWORK_INPUTS = ["signals", "labels"]
PAD_INPUTS =["signals"]

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN Covision Test')

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--log_dir', metavar='EXPORT_DIR')

    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--batch_size', metavar='N', type=int, default=1,
                        help='training and validation batch size ')

    parser.add_argument('--num_workers', metavar='N', type=int, default=2,
                        help='number of workers dataloader ')

    parser.add_argument("--log_interval", type=int, default=5)

    parser.add_argument("--val_interval", type=int, default=5)

    parser.add_argument("--model", metavar="FILE", type=str, help="Resume training from given file")

    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')

    return parser


def log_debug(msg, *args, **kwargs):
    logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    logging.get_logger().info(msg, *args, **kwargs)


def make_dataloader(args):

    print("Creating dataloaders for dataset in %s", args.data)

    # Transform
    transform = Transform(path.join(args.directory, args.data))

    # test dataloader
    test_db = CODataset(root_dir=path.join(args.directory, args.data),
                         split_name="anomaly",
                         transform=transform)

    test_sampler = BatchSampler(test_db,
                                 batch_size=args.batch_size)

    test_dl = data.DataLoader(test_db,
                               batch_sampler=test_sampler,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               num_workers=args.num_workers)

    return test_dl, transform


def make_model(args, **varargs):
    seq_len = 32
    n_features = 1
    embedding_dim = 64
    batch_size = args.batch_size
    num_layers = 1

    return LSTM_RAE(seq_len, n_features, embedding_dim, batch_size, num_layers)


def ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


def del_dir(dir_path):

    if path.isdir(dir_path):
        shutil.rmtree(dir_path)


def count_anomalies(arr, THD):
    res = 0
    anomalies = []
    positions = []

    chunk = []
    chunk_p = []

    indices = np.arange(0, len(arr)).astype(int)

    for i in range(1, len(arr)):

        j = i - 10

        if j<0:
            j=0

        if arr[i] > THD or np.mean(arr[j:i]) > THD:

            chunk = np.append(chunk, arr[i])
            chunk_p = np.append(chunk_p, indices[i])

        elif len(chunk) > 10:
            assert len(chunk) == len(chunk_p)
            anomalies.append(chunk)
            positions.append(chunk_p)

            chunk = chunk_p = []

    return anomalies, positions


def save_predictions(idxs, signals, labels, preds, out_dir=None, transform=None):

    fig = plt.figure()

    THD = 0.5

    for idx, (iid, signal_i, label_i, pred_i) in enumerate(zip(idxs, signals, labels, preds)):

        png = path.join(out_dir, str(iid) + '.png')

        signal_i, pred_i = signal_i.cpu().numpy(), pred_i.cpu().numpy()

        error = np.abs(signal_i - pred_i)

        anomalies, anomalies_idx = count_anomalies(error, THD)

        # Process gt Labels
        label_i = label_i.cpu().numpy()
        label_i = np.reshape(label_i[2: 2 + 3 * label_i[1]], (-1, 3))
        for pos in label_i:
            plt.axvline(x=pos[0], color='black', linestyle='--')
            plt.axhline(y=(pos[1] - transform["mean"]) / transform["std"], color='black', linestyle='--')

        # Process Pred Labels
        out_i = []

        for i, (err, err_idx) in enumerate(zip(anomalies, anomalies_idx)):

            start, width = int(err_idx[0]), len(err_idx)

            max_idx = np.argmax(signal_i[start: start+width], axis=0)

            position = err_idx[max_idx]
            peak = signal_i[start + max_idx]

            plt.axvline(x=position, color='blue', linestyle='--')
            plt.axhline(y=peak, color='blue', linestyle='--')

            out_i.append([position, peak, width])

        fig.suptitle('signal {} '.format(str(iid)), fontsize=10)
        fig.suptitle('Signal ({}) : detected anomalies ({}) vs gt anomalies ({})'.format(str(iid), len(anomalies),  len(label_i)), fontsize=15)

        plt.plot(signal_i, color='green', label="input signal")
        plt.plot(pred_i, color='red', label="reconstructed signal")

        plt.legend(loc='upper left', frameon=False, fontsize=5)

        fig.savefig(png, dpi=100)
        plt.cla()

    plt.close(fig)


def test(model, dataloader, **varargs):
    model.eval()
    dataloader.batch_sampler.set_epoch(0)

    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    data_time = time.time()

    losses = []
    for it, batch in enumerate(dataloader):
        with torch.no_grad():
            # Get data indices
            idxs = batch["idx"]

            # Pad batch
            for k in PAD_INPUTS:
                batch[k].pad(max_size=1024)

            # Upload batch
            batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in NETWORK_INPUTS}

            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            signals, signals_idx = batch["signals"].contiguous
            labels, labels_idx = batch["labels"].contiguous

            signal_pred = model(signals)

            loss = varargs['criterion'](signal_pred, signals)

            losses.append(loss.item())

            if dataloader.dataset.split_name == "anomaly":
                varargs['save_funtion'](idxs, signals, labels, signal_pred)

            del batch

            # Update meters
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            # Log batch
            if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
                logging.iteration(
                    None, "val", 0, 1, 1,
                    it + 1, len(dataloader),
                    OrderedDict([
                        ("loss", loss),
                        ("data_time", data_time_meter),
                        ("batch_time", batch_time_meter)
                    ])
                )

            data_time = time.time()

    return losses


def main(args):

    # Initialize
    device_id, device = args.local_rank, torch.device(args.local_rank)
    torch.cuda.set_device(device_id)

    # Initialize logging
    logging.init(args.log_dir, "testing")
    summary = tensorboard.SummaryWriter(path.join(args.directory, args.log_dir))

    # Create dataloaders
    test_dataloader, transform = make_dataloader(args)

    # Create model
    model = make_model(args)

    # Init GPU stuff
    torch.backends.cudnn.enabled = False
    model = model.cuda(device=device)

    log_debug("Loading snapshot from %s", args.model)
    resume_from_snapshot(model, args.model)

    # Metric
    criterion = torch.nn.L1Loss(reduction='mean').to(device)

    # save function

    out_dir = path.join(args.directory, "output")

    del_dir(out_dir)

    ensure_dir(out_dir)

    save_funtion = partial(save_predictions, out_dir=out_dir, transform=test_dataloader.dataset.transform)

    losses = test(model, test_dataloader,
                  save_funtion=save_funtion,
                  device=device,
                  criterion=criterion,
                  summary=summary,
                  log_interval=args.log_interval
                  )
    if test_dataloader.dataset.split_name == 'test':
        log_debug(" Get THD Value")
        sns.distplot(losses, bins=100, kde=True)
        plt.show()


if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())