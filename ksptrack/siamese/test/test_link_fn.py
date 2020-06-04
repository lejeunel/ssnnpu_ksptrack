#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

s = np.linspace(0, 1, 100)
v = 0.4
t = 2

f = s.copy()
f[s < v] = -f[s < v] / v + 1
f[s < v] = t * f[s < v]
f[s >= v] = (-f[s >= v] + 1) / (1 - v) - 1
f[s >= v] = t * f[s >= v]
f_lbt = f.copy()

# bernoulli
f_ber = -np.log(s / (1 - s))

plt.plot(s, f_lbt, 'bo-')
plt.plot(s, f_ber, 'ro-')
plt.grid()
plt.show()
