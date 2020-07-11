#!/usr/bin/env python3

from ksptrack.utils.base_dataset import BaseDataset
from skimage import exposure
from os.path import join as pjoin
import matplotlib.pyplot as plt

dl_correct = BaseDataset(pjoin('/home/ubelix/lejeune/data/medical-labeling',
                               'Dataset21'),
                         normalization='rescale',
                         resize_shape=512)
dl_default = BaseDataset(pjoin('/home/ubelix/lejeune/data/medical-labeling',
                               'Dataset21'),
                         normalization='rescale_adapthist',
                         resize_shape=512)

for sample_c, sample_d in zip(dl_correct, dl_default):
    plt.subplot(121)
    plt.imshow(sample_c['image'])
    plt.subplot(122)
    plt.imshow(sample_d['image'])
    plt.show()
