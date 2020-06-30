#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

r = x = np.linspace(0, 1, 1000)
y = r**0.05

plt.plot(x, y, 'b-')
plt.plot(x, x, 'r--')
plt.grid()
plt.show()
