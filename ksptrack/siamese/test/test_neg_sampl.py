#!/usr/bin/env python3

import torch

n_cliques = 4
n = torch.arange(4 * 100).float()
n = n.reshape(4, -1)

cliques = [torch.combinations(n_, r=2) for n_ in n]

# concatenate clique index
cliques = [
    torch.cat((edges, c * torch.ones((edges.shape[0], 1)).float()), dim=1)
    for c, edges in enumerate(cliques)
]

edges = torch.cat(cliques).T

# for each clique, generate a mask
tplts = []
for c in torch.unique(edges[-1, :]):
    cands = torch.unique(edges[:2, edges[-1, :] != c].flatten())
    idx = torch.randint(0, cands.numel(), size=((edges[-1, :] == c).sum(), ))
    tplts.append(
        torch.cat((edges[:2, edges[-1, :] == c], cands[idx][None, ...]),
                  dim=0))

tplts = torch.cat(tplts, dim=1)
print(tplts)
