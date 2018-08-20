import numpy as np
import os

#out['patches'] = np.asarray(patches)
#out['idx_i'] = idx_i
#out['idx_j'] = idx_j

root_dir = '/home/krakapwa/otlshare/medical-labeling/Dataset11/precomp_descriptors/vilar'
fname = 'vilar_im_0.npz'
path = os.path.join(root_dir,fname)

patches = np.load(path)['patches']

test = []
for i in range(len(patches)):
    #print('patch #: ' + str(i))
    if(np.any(np.isnan(patches[i]))):
       test.append(i)

print(test)
