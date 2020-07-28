#!/usr/bin/env python3

from glob import glob
from os.path import join as pjoin
import os
import numpy as np

root_path = '/home/ubelix/lejeune/data/medical-labeling'
file_ = 'flows.npz'

paths = glob(pjoin(root_path, '*', 'precomp_desc', file_), recursive=True)

for i, p in enumerate(paths):
    print('{}/{}: {}'.format(i + 1, len(paths), p))

    data = np.load(p)
    fvx = data['fvx']
    fvy = data['fvy']
    bvx = data['bvx']
    bvy = data['bvy']

    np.save(os.path.splitext(p)[0] + '_fvx.npy', fvx)
    np.save(os.path.splitext(p)[0] + '_fvy.npy', fvy)
    np.save(os.path.splitext(p)[0] + '_bvx.npy', bvx)
    np.save(os.path.splitext(p)[0] + '_bvy.npy', bvy)
