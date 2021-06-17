import numpy as np
from collections import Counter
a = np.array([1, 2, 1, 3, 3, 3, 0])
[item for item, count in Counter(a).items() if count > 1]
