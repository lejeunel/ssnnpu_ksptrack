from skimage.segmentation import find_boundaries
from scipy import io
import os
import numpy as np
import my_utils as utls
import matplotlib.pyplot as plt
import glob

f = 0

path = '/home/krakapwa/otlshare/medical-labeling/Dataset00/input-frames'

import pdb; pdb.set_trace()
ffnames = sorted(glob.glob(os.path.join(path,'*.png')))

labels_svx = io.loadmat(os.path.join(path,'svx.mat'))['labels']
labels_cntr = find_boundaries(labels_svx[...,f])
idx_cntr = np.where(labels_cntr)
im = utls.imread(ffnames[f])
im[idx_cntr[0],idx_cntr[1],:] = (255,255,255)


plt.imshow(im);plt.show()
