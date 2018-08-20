import os
import numpy as np
import learning_exp
import datetime
import feat_extr

extra_cfg = dict()

extra_cfg['feat_extr_algorithm'] = 'unet_gaze'      # set unet as feature extractor algorithm

#all_datasets = [['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03'],
#                ['Dataset20', 'Dataset21', 'Dataset22', 'Dataset23'],
#                ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'],
#                ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33']]
#all_seq_types = ['Tweezer','Slitlamp','Cochlea','Brain']

#all_datasets = ['Dataset11', 'Dataset12']
#all_datasets = ['Dataset13']
#all_seq_types = ['Cochlea']
all_datasets = ['Dataset03']
all_seq_types = ['Tweezer']
#all_datasets = ['Dataset21', 'Dataset22']
#all_datasets = ['Dataset23']
#all_seq_types = ['Slitlamp']
#all_datasets = ['Dataset33']
#all_datasets = ['Dataset31', 'Dataset32']
#all_seq_types = ['Brain']

# Run KSP on all
for d in all_datasets:
        extra_cfg['dataSetDir'] = d
        print("dset: " + d)
        for k in np.arange(1,6):
            extra_cfg['csvFileName_fg'] = 'video' + str(k) + '.csv'
            conf = feat_extr.main(extra_cfg)
