#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 1000)
ysm = np.log(1 + np.exp(x))
y = np.clip(x, a_min=0, a_max=x.max())

plt.plot(x, ysm, 'b-')
plt.plot(x, y, 'r--')
plt.grid()
plt.show()
