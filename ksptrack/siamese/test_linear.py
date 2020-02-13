
import torch
import torch.nn as nn
import torch.nn.functional as F

lin1 = nn.Linear(256, 256 // 2, bias=True)
lin2 = nn.Linear(256 // 2, 1, bias=False)

X = torch.rand(100, 256)
X = torch.stack((X, X))

out = []
out.append(F.relu(lin1(X[0, ...])))
out.append(F.relu(lin1(X[1, ...])))
out = torch.abs(out[1] - out[0])
out = lin2(out)
out = F.tanh(out)
out = 1 - out
print(out)

