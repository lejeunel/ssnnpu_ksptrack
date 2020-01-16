import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from siamese_sp.im_utils import make_grid_rag
from siamese_sp.loader import Loader
from skimage import io
import numpy as np


dl = Loader('/home/ubelix/lejeune/data/medical-labeling/Dataset30')

sample = dl[40]
probas = np.random.uniform(0, 1, size=len(sample['rag'].edges()))

im = make_grid_rag(sample['image_unnormal'],
                   sample['labels'][..., 0],
                   sample['rag'],
                   probas,
                   sample['label/segmentation'][..., 0])

io.imsave('test_plot.png', im)
