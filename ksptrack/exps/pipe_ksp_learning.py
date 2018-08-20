import os
import numpy as np
import learning_exp_unet as lexp
#import learning_exp_vgg as lexp
import yaml
import cfg_unet as cfg
#import cfg_vgg16 as cfg
import results_dirs as rd

extra_cfg = dict()

learning_root = os.path.join(rd.root_dir,
                             'learning_exps')

seq = 'Slitlamp'
out_dirs = ['learning_Slitlamp_2018-01-16_10-06-57']
#out_dirs = ['learning_Slitlamp_2018-01-11_11-35-59']
#seq = 'Brain'
#out_dirs = ['learning_Brain_2017-11-19_15-28-03']

#seq = 'Tweezer'
#out_dirs = ['learning_Tweezer_2017-11-18_16-09-20']
#out_dirs = [None]

#seq = 'Cochlea'
#out_dirs = ['learning_Cochlea_2017-11-11_17-02-18']
#out_dirs = [None]

extra_cfg = dict()
conf_train = cfg.Bunch(cfg.cfg())
extra_cfg_unet = dict()
#extra_cfg_unet['load_model'] = {'my': 'weights.h5', 'true': 'weights.h5'}
extra_cfg_unet['load_model'] = {'my': True, 'true': True}
resume_from_checkpoint = False
conf_train.__dict__.update(extra_cfg_unet)

# Run learning on sequences of same type with same gaze
gset = 0
confs = [rd.confs_dict_ksp[seq][i][gset] for i in range(4)]

if(out_dirs[0] is None):
    #learning_exp_unet.main(confs, conf_train, train=False)
    lexp.main(confs,
              conf_train)
else:
    out_dir = os.path.join(learning_root,
                           out_dirs[0])
    lexp.main(confs,
              conf_train,
              out_dir,
              train=False,
              pred=True,
              score=True,
              resume_model=extra_cfg_unet['load_model'])
