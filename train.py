import argparse
import time
import shutil



import matplotlib.pyplot as plt

from os import path
from collections import OrderedDict

import torch
import torch.utils.data as data
import torch.optim as optim

import tensorboardX as tensorboard

from net.dataset import CODataset, BatchSampler, collate_fn, preprocessing, Transform
from net.LSTM import LSTM_RAE

from utils.meters import AverageMeter
import utils.logging as logging
from utils.snapshot import save_snapshot, resume_from_snapshot


NETWORK_INPUTS = ["signals", "labels"]
PAD_INPUTS =["signals"]


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN Covision Test')

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--log_dir', metavar='EXPORT_DIR')

    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument('--batch_size', metavar='N', type=int, default=1,
                        help='training and validation batch size ')

    parser.add_argument('--num_workers', metavar='N', type=int, default=2,
                        help='number of workers dataloader ')

    parser.add_argument("--log_interval", type=int, default=5)

    parser.add_argument("--val_interval", type=int, default=5)

    parser.add_argument("--learning_rate", type=float)

    parser.add_argument("--weight_decay", type=float)

    parser.add_argument("--refresh", type=bool, default=False)

    parser.add_argument("--resume", metavar="FILE", type=str, help="Resume training from given file")

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

    if args.refresh:
        preprocessing(path.join(args.directory, args.data))

    # Transform
    transform = Transform(path.join(args.directory, args.data))


    # Training dataloader

    train_db = CODataset(root_dir=path.join(args.directory, args.data),
                         split_name="train",
                         transform=transform)

    train_sampler = BatchSampler(train_db,
                                 batch_size=args.batch_size)

    train_dl = data.DataLoader(train_db,
                               batch_sampler=train_sampler,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               num_workers=args.num_workers)

    # Validation dataloader
    val_db = CODataset(root_dir=path.join(args.directory, args.data),
                       split_name="val",
                       transform=transform)

    val_sampler = BatchSampler(val_db,
                               batch_size=args.batch_size)

    val_dl = data.DataLoader(val_db,
                             batch_sampler=val_sampler,
                             pin_memory=True,
                             collate_fn=collate_fn,
                             num_workers=args.num_workers)

    return train_dl, val_dl


def make_model(args, **varargs):
    seq_len = 32
    n_features = 1
    embedding_dim = 64
    batch_size = args.batch_size
    num_layers = 1

    return LSTM_RAE(seq_len, n_features, embedding_dim, batch_size, num_layers)


def make_optimizer(args, model):

    # Set-up optimizer hyper-parameters
    parameters = [
        {
            "params": model.parameters(),
            "lr": args.learning_rate,
        }
    ]

    # optimizer
    optimizer = optim.Adam(parameters)

    # scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    return optimizer, scheduler


def train(model, optimizer, scheduler, train_dataloader, meters, **varargs):

    model.train()
    optimizer.zero_grad()

    global_step = varargs["global_step"]

    data_time_meter = AverageMeter((), meters["loss"].momentum)
    batch_time_meter = AverageMeter((), meters["loss"].momentum)

    data_time = time.time()

    for it, batch in enumerate(train_dataloader):

        # Pad batch
        for k in PAD_INPUTS:
             batch[k].pad(max_size=1024)

        # Upload batch
        batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in NETWORK_INPUTS}

        data_time_meter.update(torch.tensor(time.time() - data_time))

        # Update scheduler
        global_step += 1
        batch_time = time.time()

        signals, signals_idx = batch["signals"].contiguous

        signal_pred = model(signals)

        loss = varargs["criterion"](signal_pred, signals)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            meters["loss"].update(loss.cpu())
            #plt.plot(signals[0].cpu().numpy())

            #plt.plot(signal_pred[0].cpu().numpy())

            #plt.show()

        batch_time_meter.update(torch.tensor(time.time() - batch_time))

        # Clean-up
        del batch, loss

        # Log
        if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
            logging.iteration(
                varargs["summary"], "train", global_step,
                varargs["epoch"] + 1, varargs["num_epochs"],
                it + 1, len(train_dataloader),
                OrderedDict([
                    ("lr", scheduler.get_lr()[0]),
                    ("loss", meters["loss"]),
                    ("data_time", data_time_meter),
                    ("batch_time", batch_time_meter)
                ])
            )

        data_time = time.time()

    return global_step


def validate(model, val_dataloader, **varargs):
    model.eval()
    val_dataloader.batch_sampler.set_epoch(varargs["epoch"])

    loss_meter = AverageMeter(())
    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    data_time = time.time()
    for it, batch in enumerate(val_dataloader):
        with torch.no_grad():

            # Pad batch
            for k in PAD_INPUTS:
                batch[k].pad(max_size=1024)

            # Upload batch
            batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in NETWORK_INPUTS}

            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            signals, signals_idx = batch["signals"].contiguous
            signal_pred = model(signals)

            loss = varargs["criterion"](signal_pred, signals)

            # Update meters
            loss_meter.update(loss.cpu())
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            del loss

            # Log batch
            if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
                logging.iteration(
                    None, "val", varargs["global_step"],
                    varargs["epoch"] + 1, varargs["num_epochs"],
                    it + 1, len(val_dataloader),
                    OrderedDict([
                        ("loss", loss_meter),
                        ("data_time", data_time_meter),
                        ("batch_time", batch_time_meter)
                    ])
                )

            data_time = time.time()

    return loss_meter.mean


def main(args):

    # Initialize
    device_id, device = args.local_rank, torch.device(args.local_rank)
    torch.cuda.set_device(device_id)

    # Initialize logging
    logging.init(args.log_dir, "training")
    summary = tensorboard.SummaryWriter(path.join(args.directory, args.log_dir))

    # Create dataloaders
    train_dataloader, val_dataloader = make_dataloader(args)

    # Create model
    model = make_model(args)

    # Init GPU stuff
    torch.backends.cudnn.enabled = False
    model = model.cuda(device=device)

    if args.resume:
        log_debug("Loading snapshot from %s", args.resume)
        snapshot = resume_from_snapshot(model, args.resume)

    # Create optimizer
    optimizer, scheduler = make_optimizer(args, model)

    #if args.resume:

    #    optimizer.load_state_dict(snapshot["state_dict"]["optimizer"])

    # criterion
    criterion = torch.nn.L1Loss(reduction='mean').to(device)
    #criterion = torch.nn.MSELoss(reduction='mean').to(device)


    # Training loop
    momentum = 1. - 1. / len(train_dataloader)
    meters = {
        "loss": AverageMeter((), momentum),
    }

    # loop params
    total_epochs = args.epochs

    if args.resume:
        starting_epoch = snapshot["training_meta"]["epoch"] +1
        best_score = snapshot["training_meta"]["best_score"]
        global_step = snapshot["training_meta"]["global_step"]

        for name, meter in meters.items():
            meter.load_state_dict(snapshot["state_dict"][name + "_meter"])
        del snapshot
    else:
        starting_epoch = 0
        best_score = 100
        global_step = 0

    for epoch in range(starting_epoch, total_epochs):
        log_info("Starting epoch %d", epoch + 1)

        scheduler.step(epoch)

        # Run training epoch
        global_step = train(model, optimizer, scheduler, train_dataloader, meters,
                            criterion=criterion,
                            device=device,
                            epoch=epoch,
                            summary=summary,
                            log_interval=args.log_interval,
                            num_epochs=total_epochs,
                            global_step=global_step)

        # Save snapshot (only on rank 0)
        snapshot_file = path.join(args.log_dir, "model_last.pth.tar")
        log_debug("Saving snapshot to %s", snapshot_file)

        meters_out_dict = {k + "_meter": v.state_dict() for k, v in meters.items()}

        save_snapshot(snapshot_file, epoch, 0, best_score, global_step,
                      model=model.state_dict(),
                      optimizer=optimizer.state_dict(),
                      **meters_out_dict)

        if (epoch + 1) % args.val_interval == 0:
            log_info("Validating epoch %d", epoch + 1)

            score = validate(model, val_dataloader,
                             criterion=criterion,
                             device=device,
                             epoch=epoch,
                             summary=summary,
                             log_interval=args.log_interval,
                             num_epochs=total_epochs,
                             global_step=global_step,
                             log_dir=args.log_dir)

            # Update the score on the last saved snapshot
            snapshot = torch.load(snapshot_file, map_location="cpu")
            snapshot["training_meta"]["last_score"] = score
            torch.save(snapshot, snapshot_file)

            del snapshot

            if score < best_score:
                best_score = score
                shutil.copy(snapshot_file, path.join(args.log_dir, "model_best.pth.tar"))


if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())