from os.path import join as pjoin
import numpy as np
from loader import Loader
import matplotlib.pyplot as plt
from ksptrack.siamese import im_utils as iutls

run_path = '/home/ubelix/lejeune/runs/siamese_dec/Dataset00'
in_root = '/home/ubelix/lejeune/data/medical-labeling/Dataset00'
init_clusters_path = pjoin(run_path, 'init_clusters.npz')

preds = np.load(init_clusters_path, allow_pickle=True)['preds']
init_clusters = np.load(init_clusters_path, allow_pickle=True)['clusters']

dl = Loader(pjoin(in_root))

frame = 10

sample = dl[frame]

plt.subplot(221)
plt.imshow(sample['image_unnormal'])
plt.subplot(222)
plt.imshow(iutls.make_clusters(sample['labels'], preds[frame]))
plt.show()
