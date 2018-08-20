import os
import vilar
import numpy as np
import learning_exp
import datetime
import yaml
import matplotlib.pyplot as plt

extra_cfg = dict()
extra_cfg['csvFileName_fg'] = 'video' + str(1) + '.csv'
extra_cfg['dataSetDir'] = 'Dataset02'
extra_cfg['calc_training_patches'] = True
extra_cfg['calc_pred_patches'] = True

root_dir = '/home/laurent.lejeune/medical-labeling'
#all_datasets=[['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03'], ['Dataset20' ,'Dataset21', 'Dataset22', 'Dataset23'],['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'], ['Dataset30' ,'Dataset31', 'Dataset32', 'Dataset33']]
#all_seq_types=['Tweezer', 'Slitlamp']
#all_datasets=['Dataset11', 'Dataset12', 'Dataset13']
#all_seq_types=['Cochlea']
#all_seq_types=['Slitlamp']
#all_datasets=['Dataset02', 'Dataset03']
all_datasets=['Dataset20', 'Dataset21', 'Dataset22', 'Dataset23']
out_dirs = [None, None]

#all_datasets=['Dataset21', 'Dataset22', 'Dataset23']
#out_dirs = ['Dataset21/results/2017-10-11_21-46-21_exp_vilar', None, None]


#all_datasets=['Dataset02', 'Dataset03']
#out_dirs = ['Dataset02/results/2017-10-06_10-27-12_exp_vilar', None]


#all_datasets=['Dataset11', 'Dataset12', 'Dataset13']
#out_dirs = [None, None, None]

#temp_folder = '/tmp'
#root_dir = '/home/laurent.lejeune/medical-labeling'
#out_dir = os.path.join(root_dir,'learning_vilar_Tweezer_2017-09-20_14-57-23')
#out_dir_self = os.path.join(root_dir, 'Dataset00', 'results', '2017-09-21_15-37-53_exp_vilar')


for i in range(len(all_datasets)):
    extra_cfg['dataSetDir'] = all_datasets[i]
    conf, clf = vilar.make_training_patches_and_fit(extra_cfg)
    conf = vilar.predict(conf, clf, n_jobs=conf.vilar_jobs)
    out_dir = conf.dataOutDir
    dir_pred_frames = conf.dataOutDir
    if(out_dirs[i] is not None):
        out_dir = os.path.join(root_dir, out_dirs[i])
        dir_pred_frames = out_dir
        extra_cfg['dataOutDir'] = out_dir
    vilar.do_exp_self_learning(out_dir=out_dir,
                               dir_pred_frames=dir_pred_frames)
