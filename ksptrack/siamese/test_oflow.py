import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from siamese_sp.im_utils import make_grid_rag
from siamese_sp.loader import Loader
from skimage import io
import numpy as np

np_file = np.load('/home/ubelix/lejeune/data/medical-labeling/Dataset00/precomp_desc/hoof.npz',
                  allow_pickle=True)
hoof = np_file['hoof'][()]
dl = Loader('/home/ubelix/lejeune/data/medical-labeling/Dataset30')
ind_grid = 0
frame = 0
labels = dl[frame]['labels'][..., 0]

mask_sp = np.zeros_like(labels)
for k in hoof['forward'][frame].keys():
    mask_sp[labels == k] = True

plt.subplot(121)
plt.imshow(labels)
plt.subplot(122)
plt.imshow(mask_sp)
plt.show()
import pdb; pdb.set_trace() ## DEBUG ##


# sample = dl[40]
# probas = np.random.uniform(0, 1, size=len(sample['rag'].edges()))

# im = make_grid_rag(sample['image_unnormal'],
#                    sample['labels'][..., 0],
#                    sample['rag'],
#                    probas,
#                    sample['label/segmentation'][..., 0])

# io.imsave('test_plot.png', im)
