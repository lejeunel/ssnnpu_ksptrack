import itertools
import SimpleITK as sitk
from medpy import io
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import my_utils as utls

root_path = '/home/laurent.lejeune/medical-labeling'

match_brats = np.load(os.path.join(root_path, 'brats_matching_1.npz'))['match'][()]
my_keys = list(match_brats.keys())

res = dict()
for k_my in my_keys:
    res[k_my] = dict()
    for k_brats in match_brats[k_my]:
        d_min = np.min(match_brats[k_my][k_brats])
        brats_min = np.argmin(match_brats[k_my][k_brats])
        res[k_my][k_brats] = d_min

# Take candidates
candidates = list()
candidates.append(sorted(res[my_keys[0]],
                   key=res[my_keys[0]].get))
candidates.append(sorted(res[my_keys[1]],
                   key=res[my_keys[1]].get))
candidates.append(sorted(res[my_keys[2]],
                   key=res[my_keys[2]].get))
candidates.append(sorted(res[my_keys[3]],
                         key=res[my_keys[3]].get))

for i in range(len(candidates)):
    print('--------------------')
    print(my_keys[i])
    print(candidates[i][0])
