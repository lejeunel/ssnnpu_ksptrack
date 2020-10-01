import torch

X = torch.tensor([[1, 2], [3, 4]])
Y = torch.tensor([[5, 6], [7, 8]])
X1 = X.unsqueeze(0)
Y1 = Y.unsqueeze(1)
print(X1.shape, Y1.shape)
X2 = X1.repeat(Y.shape[0], 1, 1)
Y2 = Y1.repeat(1, X.shape[0], 1)
print(X2.shape, X2.shape)
Z = torch.cat([X2, Y2], -1)
Z = Z.view(-1, Z.shape[-1])
print(Z.shape)
import pdb; pdb.set_trace() ## DEBUG ##
