# scheduler example
import torch
from torch.optim import lr_scheduler
from ksptrack.models.deeplab import DeepLabv3Plus
import matplotlib.pyplot as plt
import numpy as np


model = DeepLabv3Plus(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

lrs = []

for epoch in range(100):
    print(scheduler.get_lr())
    lrs.append(scheduler.get_lr())
    scheduler.step()

plt.plot(np.arange(100), lrs)
plt.grid()
plt.show()

