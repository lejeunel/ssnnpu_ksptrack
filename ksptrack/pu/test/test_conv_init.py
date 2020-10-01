#!/usr/bin/env python3

import torch

m = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
print('weight: {}'.format(m.weight.shape))
print('bias: {}'.format(m.bias.shape))

m.bias.data.fill_(0.01)

x = torch.zeros((1, 32, 128, 128))
y = m(x)

print('y: {}'.format(y))
