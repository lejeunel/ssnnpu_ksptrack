from os.path import join as pjoin
import numpy as np
from ksptrack.hoof_extractor import HOOFExtractor
from ksptrack.utils.data_manager import DataManager
from ksptrack.utils.base_dataset import BaseDataset
import matplotlib.pyplot as plt
from skimage import segmentation

def make_image(sample, r, c):
    im = sample['image']
    labels = sample['labels'][..., 0]
    label = labels[r, c]
    contour = segmentation.find_boundaries(labels == label)
    im[contour, ...] = (255, 0, 0)

    return im, labels, label

def get_hoof(df, f, l, dir_):
    df = df.loc[(df['frame'] == f) & (df['label'] == l)]['hoof_'+dir_]
    return df.to_numpy()[0]


root_path = '/home/ubelix/lejeune/data/medical-labeling/Dataset00'

dm = DataManager(root_path)
hoof_extr = HOOFExtractor(root_path, dm.desc_dir, dm.labels)
hoof_extr.make_hoof()

dset = BaseDataset(root_path)

dir_ = 'backward'
f0 = 118
f1 = f0 - 1
r, c = 200, 550

im0, labels0, l0 = make_image(dset[f0], r, c)
im1, labels1, l1 = make_image(dset[f1], r, c)
h0 = get_hoof(hoof_extr.hoof, f0, l0, dir_)
h1 = get_hoof(hoof_extr.hoof, f1, l1, dir_)

plt.subplot(221)
plt.imshow(im0)
plt.subplot(222)
plt.imshow(im1)
plt.subplot(223)
plt.stem(h0)
plt.subplot(224)
plt.stem(h1)
plt.show()

inters = np.min(np.stack((h0, h1)), axis=0).sum()
print('inters: {}'.format(inters))
