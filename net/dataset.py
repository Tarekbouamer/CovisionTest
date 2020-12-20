from os import path
import numpy as np
from matplotlib import pylab as plt

import pickle

import torch
from torch.utils.data import Dataset, Sampler

import torchvision.transforms as transforms

from utils.sequence import PackedSequence
from sklearn.preprocessing import MinMaxScaler, StandardScaler

_Ext=".pkl"


def preprocessing(dir):
    print(" load and split data")
    num_labels = 11
    bins = 1024

    # Read labels
    info_path = path.join(dir, "info.raw")
    info_data = np.fromfile(info_path, dtype='float32', sep="")
    info_data = np.reshape(info_data, (-1, num_labels))
    # np.savetxt(path.join(dir, "info.csv"),      info_data,      delimiter="  ", fmt='%s')

    # Read signals
    signal_path = path.join(dir, "signal.raw")
    signal_data = np.fromfile(signal_path, dtype='uint8', sep="")
    signal_data = np.reshape(signal_data, (-1, bins))
    # np.savetxt(path.join(dir, "signal.csv"),    signal_data,    delimiter="  ", fmt='%s')

    assert len(info_data) == len(signal_data), "data  unbalanced, cannot perform split "

    # Extract normal data set
    indices = np.arange(0, len(info_data))

    normal_msk = info_data[:, 1] == 0.0

    normal_signal = signal_data[normal_msk]
    normal_label = info_data[normal_msk]
    normal_idx = indices[normal_msk]

    anomaly_signal = signal_data[~normal_msk]
    anomaly_label = info_data[~normal_msk]
    anomaly_idx = indices[~normal_msk]

    # Split Train, Val, Tes
    normal_indices = np.arange(0, len(normal_signal))

    np.random.shuffle(normal_indices)

    edge1, edge2 = round(len(normal_indices) * 0.70), round(len(normal_indices) * 0.85)

    train_idx, val_idx, test_idx = normal_indices[:edge1], normal_indices[edge1:edge2], normal_indices[edge2:]

    # Stats
    data = normal_signal[train_idx].reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit_transform(data)

    Stdscaler = StandardScaler()
    Stdscaler.fit_transform(data)

    train_db = {
        'labels': normal_label[train_idx],
        'signals': normal_signal[train_idx],
        'idx': normal_idx[train_idx]
    }

    val_db = {
        'labels': normal_label[val_idx],
        'signals': normal_signal[val_idx],
        'idx': normal_idx[val_idx]
    }

    test_db = {
        'labels': normal_label[test_idx],
        'signals': normal_signal[test_idx],
        'idx': normal_idx[test_idx]
    }

    stats = {
        'mean': Stdscaler.mean_[0],
        'std': Stdscaler.scale_[0],
        'max': scaler.data_max_[0],
        'min': scaler.data_min_[0]
    }
    print(stats)

    # Save normal splits
    pickle.dump(train_db,   open(path.join(dir, "train.pkl" ),  "wb"))
    pickle.dump(val_db,     open(path.join(dir, "val.pkl"   ),  "wb"))
    pickle.dump(test_db,     open(path.join(dir, "test.pkl"   ),  "wb"))
    pickle.dump(stats,      open(path.join(dir, "stats.pkl"   ),  "wb"))

    # Save anomalies
    anomaly_db = {
        'labels': anomaly_label,
        'signals': anomaly_signal,
        'idx': anomaly_idx
    }

    pickle.dump(anomaly_db, open(path.join(dir, "anomaly.pkl"), "wb"))

    print(" >> Done ...")


class CODataset(Dataset):

    def __init__(self, root_dir, split_name, transform=None, visualization=False):

        self.root_dir = root_dir
        self.split_name = split_name

        self.transform = transform


        self.database = pickle.load(open(path.join(self.root_dir, self.split_name + _Ext), "rb"))

        if visualization:
            self.visualization()

    def visualization(self):

        num_subplots = 3

        fig, axs = plt.subplots(num_subplots, 1, constrained_layout=True)

        for k in range(0, num_subplots):
            i = round(np.random.uniform(0, len(self.database['train_label']) -1))

            id = self.database['train_idx'][i]
            train_signal = self.database['train_signal'][i]

            train_label = self.database['train_label'][i]
            last = int(2 + 3 * train_label[1])

            data = np.reshape(train_label[2:last], (-1, 3))
            for p in data:

                axs[k].axvline(x=p[0], color='r', linestyle='--')
                axs[k].axhline(y=p[1], color='r', linestyle='--')
                axs[k].axhline(y=p[1], color='r', linestyle='--')

            axs[k].plot(train_signal)
            axs[k].set_title('subplot of signal {} '.format(id))

        plt.show()

    def __len__(self):
        return len(self.database['labels'])

    def __getitem__(self, item):

        # to tensor
        #signals = (self.database["signals"][item] - self.transform['min']) / (self.transform['max'] - self.transform['min'])
        signals = (self.database["signals"][item] - self.transform['mean']) / (self.transform['std'])

        signals = torch.from_numpy(signals).float()
        labels = torch.from_numpy(self.database["labels"][item]).long()

        idx = self.database["idx"][item]

        return dict(signals=signals, labels=labels, idx=idx)


class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size=1, drop_last=False, epoch=0):
        super(BatchSampler, self).__init__(data_source)

        self.data = data_source.database

        self.batch_size = batch_size
        self.drop_last = drop_last
        self._epoch = epoch

        self.set = np.arange(len(self.data["idx"]))

    def _generate_batches(self):
        np.random.shuffle(self.set)

        batches = []
        batch = []

        for id in self.set:
            batch.append(id)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        return batches

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.set) // self.batch_size

    def __iter__(self):

        # get all batches
        batches = self._generate_batches()

        g = torch.Generator()
        g.manual_seed(self._epoch)

        # generate indices for batches
        indices = list(torch.arange(len(batches)))

        assert len(indices) == self.__len__()

        for idx in indices:
            batch = sorted(batches[idx])
            yield batch


def Transform(dir):

    stats = pickle.load(open(path.join(dir, "stats" + _Ext), "rb"))

    normalize = transforms.Normalize(mean=stats['mean'], std=stats['std'])

    transform = transforms.Compose([transforms.ToTensor(), normalize, ])

    return stats


def collate_fn(items):
    out = {}
    if len(items) > 0:
        for key in items[0]:
            out[key] = [item[key] for item in items]
            if isinstance(items[0][key], torch.Tensor):
                out[key] = PackedSequence(out[key])
    return out


