#!/usr/bin/env python3

from ksptrack.utils import pb_hierarchy_extractor as pbh
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from torch.utils.data import DataLoader
# from ksptrack.siamese.modeling.siamese import Siamese
# from ksptrack.siamese import utils
import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin
from skimage import io
import numpy as np
from ksptrack.siamese.tree_explorer import TreeExplorer

data_dir = '/home/ubelix/lejeune/data/medical-labeling/Dataset00'
# data_dir = '/home/ubelix/lejeune/data/medical-labeling/Dataset30'
pbex = pbh.PbHierarchyExtractor(data_dir)

f = 20

# dset = LocPriorDataset(root_path=data_dir, normalization='rescale')

# dl = DataLoader(dset, collate_fn=dset.collate_fn)

# device = torch.device('cuda')
# model = Siamese(embedded_dims=15, cluster_number=15, siamese='none').to(device)
# cp = pjoin('/home/ubelix/artorg/lejeune/runs/siamese_dec/Dataset00',
#            'checkpoints/cp_pu.pth.tar')
# state_dict = torch.load(cp, map_location=lambda storage, loc: storage)
# model.load_state_dict(state_dict)
# model.eval()

# for data in dl:
#     data = utils.batch_to_device(data, device)
#     rho = model(data)['rho_hat'].sigmoid().detach().cpu().numpy().squeeze()
#     io.imsave(pjoin(data_dir, 'rho.png'), rho)

rho = io.imread(pjoin(data_dir, 'rho.png')) / 255
data = pbex[f]
# proped = pbh.propagate_weights(s['tree'], s['labels'], rho)
# mapping = pbh.get_cut_thr(s['tree'], proped, thr=0.2)
# mapping = pbh.labelisation_optimal_cut(s['tree'], s['labels'], rho)
# new_labels = pbh.relabel(s['labels'], mapping)

kps = data['loc_keypoints']
texp = TreeExplorer(data['tree'],
                    data['labels'],
                    rho,
                    np.fliplr(kps.to_xy_array()).astype(int).tolist(),
                    sort_order='descending')

level0 = 0
level1 = 4
merged_label0 = np.array([data['labels'] == n
                          for n in texp[level0]['nodes']]).sum(axis=0)
merged_label1 = np.array([data['labels'] == n
                          for n in texp[level1]['nodes']]).sum(axis=0)
print(texp[level0])
print(texp[level1])

im = kps.draw_on_image(data['image'], size=7)
plt.subplot(321)
plt.imshow(data['labels'])
plt.subplot(322)
plt.imshow(im)
plt.subplot(323)
plt.imshow(rho)
plt.subplot(324)
plt.imshow(data['pb'])
plt.subplot(325)
plt.imshow(merged_label0)
plt.subplot(326)
plt.imshow(merged_label1)
plt.show()
import pdb
pdb.set_trace()  ## DEBUG ##
