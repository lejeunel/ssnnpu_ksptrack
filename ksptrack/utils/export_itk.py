import SimpleITK as sitk
import sys
import os
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

# file_ = '/home/krakapwa/otlshare/laurent.lejeune/Downloads/BRATS2015_Training/HGG/brats_2013_pat0008_1/VSD.Brain_3more.XX.XX.OT.54559/VSD.Brain_3more.XX.XX.OT.54559.mha'

# file_ = '/home/krakapwa/otlshare/laurent.lejeune/Downloads/medical_segm_decathlon/Task07_Pancreas/Task07_Pancreas/imagesTr/pancreas_421.nii.gz'

# file_ = '/home/krakapwa/otlshare/laurent.lejeune/Downloads/medical_segm_decathlon/Task07_Pancreas/Task07_Pancreas/labelsTr/pancreas_421.nii.gz'

# file_ = '/home/laurent.lejeune/Downloads/medical_segm_decathlon/Task03_Liver/Task03_Liver/imagesTr/liver_0.nii.gz'

file_ = '/home/laurent.lejeune/Downloads/medical_segm_decathlon/Task03_Liver/Task03_Liver/labelsTr/liver_0.nii.gz'

out_shape = (700, 700)

path = os.path.split(file_)[0]

img = sitk.ReadImage(file_)
nda = sitk.GetArrayFromImage(img)

for i in range(nda.shape[0]):
    fname = os.path.join(path, 'frame_{0:04d}.png'.format(i))
    im = nda[i, ...]
    im = transform.resize(im,
                          out_shape,
                          preserve_range=True,
                          mode='reflect').astype(int)
    min_ = im.min()
    max_ = im.max()
    range_ = max_ - min_
    im = (im - min_) / range_

    if len(im.shape) < 3:
        im = np.repeat(im[..., np.newaxis], 3, axis=-1)

    print('Saving to {}'.format(fname))
    io.imsave(os.path.join(path, fname), im)
