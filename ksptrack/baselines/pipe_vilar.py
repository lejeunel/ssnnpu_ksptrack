import os
import vilar
import numpy as np
import learning_exp
import datetime
import yaml
import matplotlib.pyplot as plt
import itertools
import pickle as pk

extra_cfg = dict()
extra_cfg['csvFileName_fg'] = 'video' + str(1) + '.csv'
extra_cfg['calc_training_patches'] = True
extra_cfg['calc_pred_patches'] = True


root_dir = '/home/laurent.lejeune/medical-labeling'
#datasets=['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13']
#datasets=['Dataset20', 'Dataset21', 'Dataset22', 'Dataset23']
#datasets=['Dataset22', 'Dataset23']
#datasets=['Dataset00','Dataset01','Dataset02','Dataset03',
#          'Dataset10','Dataset11','Dataset12','Dataset13',
#          'Dataset20','Dataset21','Dataset22',
#          'Dataset30','Dataset31','Dataset32','Dataset33']
#
#out_dirs = [
#    'Dataset00/results/2017-10-03_08-44-26_exp_vilar',
#    'Dataset01/results/2017-10-04_14-41-12_exp_vilar',
#    'Dataset02/results/2017-10-27_15-14-21_exp_vilar',
#    'Dataset03/results/2017-10-29_12-31-45_exp_vilar',
#    'Dataset10/results/2017-10-27_15-16-57_exp_vilar',
#    'Dataset11/results/2017-10-29_02-56-42_exp_vilar',
#    'Dataset12/results/2017-10-30_11-35-51_exp_vilar',
#    'Dataset13/results/2017-10-31_14-06-19_exp_vilar',
#    'Dataset20/results/2017-10-27_15-17-35_exp_vilar',
#    'Dataset21/results/2017-10-29_03-04-10_exp_vilar',
#    'Dataset22/results/2017-11-02_09-28-37_exp_vilar',
#    'Dataset30/results/2017-10-27_15-17-50_exp_vilar',
#    'Dataset31/results/2017-11-02_11-36-54_exp_vilar',
#    'Dataset32/results/2017-10-29_15-34-05_exp_vilar',
#    'Dataset33/results/2017-11-03_09-06-38_exp_vilar']

datasets = ['Dataset12']
#out_dirs = ['Dataset32/results/2017-10-29_15-34-05_exp_vilar']
out_dirs = [None]
for i in range(len(datasets)):
    extra_cfg['ds_dir'] = datasets[i]
    extra_cfg['gamma'] = 0.0005
    extra_cfg['C'] = 100
    if(extra_cfg['calc_training_patches']):
        if(out_dirs[i] is None):
            conf, clf = vilar.make_training_patches_and_fit(extra_cfg)
            conf = vilar.predict(conf,
                                clf,
                                n_jobs=conf.vilar_jobs)
        if(out_dirs[i] is not None):
            conf, clf = vilar.make_training_patches_and_fit(extra_cfg,
                                                            os.path.join(root_dir,
                                                                            out_dirs[i]))
            conf = vilar.predict(conf,
                                clf,
                                n_jobs=conf.vilar_jobs)
    else:

        with open(os.path.join(root_dir,out_dirs[i],'cfg.yml'), 'r') as outfile:
            conf = yaml.load(outfile)
        #with open(os.path.join(root_dir,out_dirs[i],'clf.p'), 'rb') as outfile:
        #    clf = pk.load(outfile)

    vilar.calc_score(conf)
