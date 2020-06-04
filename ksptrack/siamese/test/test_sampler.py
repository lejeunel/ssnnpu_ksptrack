#!/usr/bin/env python3
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler
from torch.utils import data
from torch._six import int_classes as _int_classes
import torch


class DummyDataset(data.Dataset):
    """
    Loads and augments images and ground truths
    """
    def __init__(self, size=100):

        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'idx': idx,
            'tensor': torch.rand(5, 5),
        }

    @staticmethod
    def collate_fn(data):
        return {
            'idx': [s['idx'] for s in data],
            'tensor': torch.cat([s['tensor'][None, ...] for s in data])
        }


class RandomBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    def __init__(self, size, batch_size):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size
        self.size = size

    def __iter__(self):
        batch = []
        starts = list(range(self.size - (self.batch_size - 1)))
        # random.shuffle(starts)
        for start in starts:
            for i in range(self.batch_size):
                batch.append(start + i)
            yield batch
            batch = []

    def __len__(self):
        return self.size - (self.batch_size - 1)


size = 74
batch_sampler = RandomBatchSampler(size, batch_size=2)
print(list(batch_sampler))
dset = DummyDataset(size)
dl = DataLoader(dset, batch_sampler=batch_sampler, collate_fn=dset.collate_fn)

for i, s in enumerate(dl):
    print('i: {}, frame_idx: {}'.format(i, s['idx']))
