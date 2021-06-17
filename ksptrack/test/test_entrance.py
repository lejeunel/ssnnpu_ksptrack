import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
import pickle as pk
from os.path import join as pjoin

res_path = '/home/ubelix/lejeune/runs/ksptrack/Dataset00/aapu'
npf = np.load(pjoin(res_path, 'results.npz'), allow_pickle=True)
paths_for = npf['paths_for']
paths_back = npf['paths_back']

print(paths_for)
# print(paths_back)

root = '/home/ubelix/lejeune/data/medical-labeling/Dataset00'
f0 = 43
f1 = 44
dl = LocPriorDataset(root)

look_for_0 = 256
look_for_1 = 261
# look for
res = []
for p in paths_for:
    if ((f0, look_for_0) in p):
        res.append(p)

s0 = dl[f0]
s1 = dl[f1]
labels0 = np.squeeze(s0['labels'])
labels1 = np.squeeze(s1['labels'])

plt.subplot(221)
plt.imshow(s0['image'])
plt.subplot(222)
plt.imshow(labels0)
plt.subplot(223)
plt.imshow(s1['image'])
plt.subplot(224)
plt.imshow(labels1)
plt.show()

file_graph = pjoin(root, 'precomp_desc', 'hoof_inters_graph.npz')
# file_graph = pjoin(root, 'precomp_desc', 'transition_constraint.p')
print('loading graph...')
with open(file_graph, 'rb') as f:
    g = pk.load(f)
print('done')
import pdb
pdb.set_trace()  ## DEBUG ##

plt.subplot(121)
plt.imshow(s0['image'])
plt.subplot(122)
plt.imshow(labels0)
plt.show()

# t0 = s0['label_keypoints'][0]
t0 = look_for_0
l0 = labels0 == t0

dists = [(k[1], v['dist']) for k, v in g[(f0, t0)].items() if k[0] == f1]
inters = [(k[1], v['forward']) for k, v in g[(f0, t0)].items() if k[0] == f1]
dist_map = np.zeros(l0.shape)
inters_map = np.zeros(l0.shape)

print(dists)

for dist in dists:
    dist_map[labels1 == dist[0]] = dist[1]

for inter in inters:
    inters_map[labels1 == inter[0]] = inter[1]

import pdb
pdb.set_trace()  ## DEBUG ##
image = s0['image']
image[l0[..., 0], ...] = (255, 0, 0)
plt.subplot(221)
plt.imshow(image)
plt.title('image')
plt.subplot(222)
plt.imshow(inters_map)
plt.title('hoof inter')
plt.subplot(223)
plt.imshow(dist_map)
plt.title('dist map')
plt.show()
