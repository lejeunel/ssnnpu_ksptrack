#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

alpha = 3
p = np.linspace(0, 1, 100)
p_ = p.copy()**alpha

# bernoulli
f_ber = -np.log(p / (1 - p))
f_alpha = -np.log(p_ / (1 - p_))

plt.subplot(211)
plt.plot(p, p, 'bo-')
plt.plot(p, p_, 'ro-')
plt.grid()
plt.subplot(212)
plt.plot(p, f_ber, 'bo-')
plt.plot(p, f_alpha, 'ro-')
plt.grid()
plt.show()
