import os
from ksptrack import iterative_ksp
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import write_frames_results as writef
from cfgs.params import get_params

root_dir = '/home/laurent.lejeune/medical-labeling'

in_dirs = [
    [
        'Dataset00',
        'Dataset01',
        'Dataset02',
        'Dataset03',
        'Dataset04',
        'Dataset05'],
    [
    'Dataset10',
    'Dataset11',
    'Dataset12',
    'Dataset13'
    ],
    ['Dataset20',
    'Dataset21',
    'Dataset22',
    'Dataset23',
    'Dataset24',
    'Dataset25'],
    ['Dataset30',
    'Dataset31',
    'Dataset32',
    'Dataset33',
    'Dataset34']
]

entrance_masks = [
    'Dataset00_2019-04-10_11-38-11',
    'Dataset10_2019-05-06_11-44-46',
    'Dataset10_2019-05-06_11-44-46',
    'Dataset30_2019-05-06_09-59-33',
]

for i in range(len(in_dirs)):
    for j in range(len(in_dirs[i])):
        # with fixed radius
        os.system('python single_ksp_gpu.py \
        --cuda \
        --in-path {}/{} \
        --out-path {}/{}'.format(
            root_dir,
            in_dirs[i][j],
            root_dir,
            in_dirs[i][j]))
        # with entrance masks
        os.system('python single_ksp_gpu.py \
        --cuda \
        --in-path {}/{} \
        --out-path {}/{} \
        --entrance-masks-path {}/unet_region/runs/{}/{}/entrance_masks'.format(
            root_dir,
            in_dirs[i][j],
            root_dir,
            in_dirs[i][j],
            root_dir,
            entrance_masks[i],
            in_dirs[i][j]))


