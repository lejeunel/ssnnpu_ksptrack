import torch
from torch import nn


loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)*2
target = torch.empty(3).random_(2)
output = loss(input, target)
output.backward()
