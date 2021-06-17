import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

root_pt = '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling/Dataset24/precomp_descriptors'
feats_pt = os.path.join(root_pt, 'sp_desc_ung.p')
feats_maps_pt = os.path.join(root_pt, 'feats_unet.npz')

feats_keras = '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling/Dataset23/precomp_descriptors/sp_desc_ung.p'

pt_df = pd.read_pickle(feats_pt)
# keras_df = pd.read_pickle(feats_keras)

pt_maps = np.load(feats_maps_pt)['feats']

df = pt_df

idx0 = 0
idx1 = 12300
plt.subplot(221); plt.stem(df['desc'].loc[idx0])
plt.title('idx0')
plt.subplot(222); plt.stem(df['desc'].loc[idx1])
plt.title('idx1')
diff = df['desc'].loc[idx0] - df['desc'].loc[idx1]
plt.subplot(223); plt.stem(diff)
plt.title('idx0 - idx1')
plt.show()
# plt.subplot(121)
# plt.imshow(pt_maps[80, 10, ...])
# plt.subplot(122)
# plt.imshow(pt_maps[40, 10, ...])
# plt.show()
