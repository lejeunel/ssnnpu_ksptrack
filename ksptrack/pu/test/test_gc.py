#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import maxflow
from skimage.filters import threshold_otsu
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

npz = np.load(
    '/home/ubelix/lejeune/Documents/software/ksptrack/ksptrack/pu/gc_data_br.npz',
    allow_pickle=True)
probas = npz['probas_map']
image = npz['image']
labels = npz['labels']
probas_sp = npz['probas'][0]

init_pi = 0.235

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
ax = ax.flatten()

im = ax[1].imshow(probas)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax[1].set_title('probas')
ax[1].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
vals, bins, _ = ax[0].hist(probas_sp.ravel(), range=(0, 1), bins=100)
thr = threshold_otsu(probas_sp)
thr_mean = probas.mean()
thr_fix = 0.5

ax[0].axvline(thr, color='g', label='2-class Otsu')
ax[0].axvline(thr_fix, color='r', label='fixed thr: {}'.format(thr_fix))
ax[0].axvline(thr_mean, color='m', label='mean')
ax[0].legend()

map_ = probas >= thr
rho = map_.sum() / map_.size
ax[2].imshow(map_)
ax[2].set_title('probas >= {:.2f} [rho={:2f}]'.format(thr, rho))
ax[2].axis('off')

map_ = probas >= thr_mean
rho = map_.sum() / map_.size
ax[3].imshow(probas >= thr_mean)
ax[3].set_title('probas >= {:.2f} [rho={:2f}]'.format(thr_mean, rho))
ax[3].axis('off')

map_ = probas >= thr_fix
rho = map_.sum() / map_.size
ax[4].imshow(probas >= thr_fix)
ax[4].set_title('probas >= {:.2f} [rho={:2f}]'.format(thr_fix, rho))
ax[4].axis('off')

ax[5].axis('off')

plt.tight_layout()

plt.show()
