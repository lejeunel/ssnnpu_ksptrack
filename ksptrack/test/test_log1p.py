#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 100)
y = np.log(1 + np.exp(x))
plt.plot(x, y)
plt.grid()
plt.show()
