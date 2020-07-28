#!/usr/bin/env python3

from glob import glob
from os.path import join as pjoin
import os
import numpy as np

root_path = '/home/ubelix/lejeune/data/medical-labeling'
file_ = 'sp_labels.npz'
field = 'sp_labels'

paths = glob(pjoin(root_path, '*', 'precomp_desc', file_), recursive=True)

print(paths)

import pdb
pdb.set_trace()  ## DEBUG ##
for p in paths:
    arr = np.load(p)[field]
    arr = np.moveaxis(arr, -1, 0)
    new_path = pjoin(os.path.splitext(p)[0] + '.npy')
    np.save(new_path, arr)
