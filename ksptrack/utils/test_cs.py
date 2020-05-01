#!/usr/bin/env python3

import numpy as np

# v0 = np.random.randn(100)
v0 = np.array(100 * [1])
v0 = v0 / np.linalg.norm(v0)
# v1 = np.random.randn(100)
v1 = np.array(100 * [-1])
v1 = v1 / np.linalg.norm(v1)

dot = np.dot(v0, v1)
print(dot)
