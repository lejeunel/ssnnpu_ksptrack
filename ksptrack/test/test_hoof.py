from os.path import join as pjoin
import numpy as np
from ksptrack.hoof_extractor import HOOFExtractor
from ksptrack.utils.base_dataset import BaseDataset
import matplotlib.pyplot as plt
from skimage import segmentation


def make_image(sample, r, c):
    im = sample['image']
    labels = sample['labels']
    label = labels[r, c]
    contour = segmentation.find_boundaries(labels == label)
    im[contour, ...] = (255, 0, 0)

    return im, labels, label


def get_hoof(df, f, l, dir_):
    df = df.loc[(df['frame'] == f) & (df['label'] == l)]['hoof_' + dir_]
    return df.to_numpy()[0]


root_path = '/home/ubelix/lejeune/data/medical-labeling/Dataset11'

dset = BaseDataset(root_path)
hoof_extr = HOOFExtractor(root_path, dset.labels)
hoof_extr.make_hoof()

dir_ = 'backward'
f0 = 50
f1 = f0 - 1
r, c = 200, 450

im0, labels0, l0 = make_image(dset[f0], r, c)
im1, labels1, l1 = make_image(dset[f1], r, c)
h0 = get_hoof(hoof_extr.hoof, f0, l0, dir_)
h1 = get_hoof(hoof_extr.hoof, f1, l1, dir_)
inters = np.min(np.stack((h0, h1)), axis=0).sum()

plt.subplot(221)
plt.imshow(im0)
plt.subplot(222)
plt.imshow(im1)
plt.subplot(223)
plt.stem(h0)
plt.subplot(224)
plt.stem(h1)
plt.title('inters: {}'.format(inters))
plt.show()
