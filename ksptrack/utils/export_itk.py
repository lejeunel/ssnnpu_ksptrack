import SimpleITK as sitk
import sys
import os
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

mha_file = '/home/krakapwa/otlshare/laurent.lejeune/Downloads/BRATS2015_Training/HGG/brats_2013_pat0008_1/VSD.Brain_3more.XX.XX.OT.54559/VSD.Brain_3more.XX.XX.OT.54559.mha'

out_shape = (700, 700)

path = os.path.split(mha_file)[0]

img = sitk.ReadImage(mha_file)
nda = sitk.GetArrayFromImage(img)

for i in range(nda.shape[0]):
    fname = os.path.join(path, 'frame_{0:04d}.png'.format(i))
    im = nda[i, ...]
    im = transform.resize(im,
                          out_shape,
                          preserve_range=True,
                          mode='reflect').astype(int)
    im = im / np.max(im)
    print('Saving to {}'.format(fname))
    io.imsave(os.path.join(path, fname), im)
