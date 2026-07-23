#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: utils.py
# --- Creation Date: 24-02-2020
# --- Last Modified: Mon 24 Feb 2020 04:25:01 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Utils for VP metrics
"""

import os
import torch
import numpy as np
import torchvision
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))


def fraction_count(total, fraction):
    """Floor a fractional count without binary floating-point underflow."""
    return int(total * fraction + 1e-9)


def make_fold_split(n_data, train_fraction, fold, n_folds, seed):
    """Create cyclic folds whose validation sets overlap as little as possible."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= fold < n_folds:
        raise ValueError("fold must be in [0, n_folds)")

    permutation = np.random.RandomState(seed).permutation(n_data)
    validation_size = fraction_count(n_data, 1.0 - train_fraction)
    start = (fold * n_data) // n_folds
    validation_positions = (start + np.arange(validation_size)) % n_data
    train_size = n_data - validation_size
    train_positions = (start + validation_size + np.arange(train_size)) % n_data
    return permutation[train_positions], permutation[validation_positions]


class EpochSubsetSampler(torch.utils.data.Sampler):
    """Rotate a fixed-size epoch budget through a larger training pool."""

    def __init__(self, indices, samples_per_epoch, seed):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.samples_per_epoch = int(samples_per_epoch)
        if not 0 < self.samples_per_epoch <= len(self.indices):
            raise ValueError("samples_per_epoch must be in [1, len(indices)]")
        self.seed = seed
        self.epoch = 0
        self._cycle = np.random.RandomState(seed).permutation(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        count = len(self._cycle)
        start = (self.epoch * self.samples_per_epoch) % count
        positions = (start + np.arange(self.samples_per_epoch)) % count
        selected = self._cycle[positions].copy()
        np.random.RandomState(self.seed + self.epoch + 1).shuffle(selected)
        return iter(selected.tolist())

    def __len__(self):
        return self.samples_per_epoch


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def show_inputs_target(inputs, target, result_dir):
    img = torchvision.utils.make_grid(inputs)
    img = img / 2 + 0.5  # unnormalize
    img_np = img.numpy()
    img_np = (np.transpose(img_np, (1, 2, 0)) * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(os.path.join(result_dir, 'sainity.jpg'))
    print('labels:', str(target))
