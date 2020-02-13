import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from ksptrack.utils.base_dataset import BaseDataset


root_path = '/home/ubelix/lejeune/data/medical-labeling/'
dset_dir = 'Dataset00'
desc_dir = 'precomp_desc'
frame = 91

dset = BaseDataset(pjoin(root_path, dset_dir))

labels = np.load(pjoin(root_path, dset_dir, desc_dir, 'sp_labels.npz'))['sp_labels']
plt.subplot(121)
plt.imshow(labels[..., frame] == 406)
plt.subplot(122)
plt.imshow(dset[frame]['image'])
plt.show()
