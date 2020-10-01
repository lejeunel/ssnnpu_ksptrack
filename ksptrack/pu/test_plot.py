import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from skimage import io
import numpy as np


# dl = Loader('/home/ubelix/lejeune/data/medical-labeling/Dataset10')
dl = LocPriorDataset(root_path='/home/ubelix/lejeune/data/medical-labeling/Dataset00',
                     normalization='std',
                     resize_shape=512,
                     csv_fname='video2.csv')

plt.ion()
sample = dl[40]
plt.subplot(121)
plt.imshow(sample['image_unnormal'])
plt.subplot(122)
plt.imshow(np.squeeze(sample['labels']))
plt.show()
import pdb; pdb.set_trace() ## DEBUG ##
