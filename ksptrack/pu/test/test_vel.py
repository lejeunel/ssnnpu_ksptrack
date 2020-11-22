#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# initial state
pi0 = 0.23

# initial step size
v_fac_start = 0.005
v_fac_end = 0.05

# n epochs
T = 100

v_start = pi0 * v_fac_start
v_end = pi0 * v_fac_end

t = np.linspace(0, 100, 1000)
v = v_start + (v_end - v_start) * t / T
# pi = pi0 - (v0 + a * t)

plt.plot(t, v)
plt.grid()
plt.show()
