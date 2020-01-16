from os.path import join as pjoin
import numpy as np
from ksptrack.regionprops import regionprops
import matplotlib.pyplot as plt


frame = 1

path = '/home/laurent/Desktop/precomp_desc'
labels = np.load(pjoin(path, 'sp_labels.npz'))['sp_labels'][..., frame]
flows = np.load(pjoin(path, 'flows.npz'))
frame = 1

bvx = flows['bvx'][..., frame]
bvy = flows['bvy'][..., frame]
fvx = flows['fvx'][..., frame]
fvy = flows['fvy'][..., frame]

regions = regionprops(labels,
                      forward_flow=np.stack((fvx, fvy)),
                      backward_flow=np.stack((bvx, bvy)))

centroids = [p.centroid for p in regions]
hoof = [p.forward_hoof for p in regions]

print('got {} labels'.format(np.unique(labels)))
print('got {} centroids'.format(len(centroids)))
print('got {} hoof'.format(len(hoof)))
