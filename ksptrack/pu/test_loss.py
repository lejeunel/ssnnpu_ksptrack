from torch import nn
import torch


m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2) - 0.2
output = loss(m(input), target)
output.backward()
print(input)
print(target)
print(output)
