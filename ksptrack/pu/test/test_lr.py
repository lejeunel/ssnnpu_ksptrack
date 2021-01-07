#!/usr/bin/env python3
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt

lr_start = 1e-4
lr_end = 1e-6
n_epc = 100

tnsr = torch.randn(100)

gamma_lr = np.exp((1 / n_epc) * np.log(lr_end / lr_start))

optimizer = optim.Adam([tnsr], lr=lr_start)
lr_sch = torch.optim.lr_scheduler.StepLR(optimizer,
                                         step_size=1,
                                         gamma=gamma_lr)

lrs = []
for e in range(n_epc):

    lrs.append(lr_sch.get_last_lr()[0])
    print('e: {}/{}, lr: {}'.format(e + 1, n_epc, lrs[-1]))
    lr_sch.step()

plt.plot(np.arange(n_epc), lrs)
plt.grid()
plt.show()
