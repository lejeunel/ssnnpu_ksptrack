import os
import iterative_ksp
import test_trans_costs
import test_unet_output
import numpy as np
import datetime
import cfg_unet
import yaml
import numpy as np
import matplotlib.pyplot as plt
import glob
import my_utils as utls
import pandas as pd

all_datasets = ['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03',
                'Dataset10', 'Dataset11', 'Dataset12', 'Dataset13',
                'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23',
                'Dataset30', 'Dataset31', 'Dataset32', 'Dataset33']
root_dir = '/home/laurent.lejeune/medical-labeling/'
framePrefix = 'frame_'
frameDigits = 4
frameDir = 'input-frames'
frameExtension = '.png'

overfeat_pdname = 'overfeat.p'

for i in range(len(all_datasets)):
    print('Converting ' + all_datasets[i])
    overfeat_path = os.path.join(root_dir, all_datasets[i], 'EE', 'overfeat_wide')
    overfeat_files = sorted(glob.glob(os.path.join(overfeat_path, 'frame*.npz')))
    frame_filenames =  utls.makeFrameFileNames(framePrefix,
                                               frameDigits,
                                               frameDir,
                                               root_dir,
                                               all_datasets[i],
                                               frameExtension)
    feats = []
    for j in range(len(overfeat_files)):
        print('File {}/{}'.format(j+1, len(overfeat_files)))
        feat = np.load(overfeat_files[j])['features']
        for l in range(feat.shape[0]):
            feats.append((j, int(feat[l,0]), feat[l,1:]))

    feat_pd = pd.DataFrame(feats, columns = ['frame', 'sp_label', 'desc'] )

    feat_pd.to_pickle(os.path.join(root_dir, all_datasets[i],
                                   'precomp_descriptors',
                                   overfeat_pdname))
