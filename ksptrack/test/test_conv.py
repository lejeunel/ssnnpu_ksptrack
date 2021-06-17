#!/usr/bin/env python3

from torch import nn


def print_n_params(layer):
    n_weights = layer.weight.numel()
    n_bias = layer.bias.numel()
    print('n_weights: {}'.format(n_weights))
    print('n_bias: {}'.format(n_bias))
    print('total: {}'.format(n_bias + n_weights))


print('convolutional')
conv = nn.Conv2d(1, 6, 1, 1)
print_n_params(conv)

print('linear')
conv = nn.Linear(1, 6)
print_n_params(conv)
