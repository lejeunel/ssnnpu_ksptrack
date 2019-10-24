from ruamel import yaml
import os
from collections import OrderedDict
import copy
import glob

# Paths and frames/gaze priorities for plots
best_folds_learning = dict()

root_dir = '/home/laurent.lejeune/medical-labeling'


def datasetdir_to_type(dir_):
    """ Get sequence type from directory name
    """

    if ((dir_ == 'Dataset00') or (dir_ == 'Dataset01') or (dir_ == 'Dataset02')
            or (dir_ == 'Dataset03')):
        seq_type = 'tweezer'
    elif ((dir_ == 'Dataset30') or (dir_ == 'Dataset31')
          or (dir_ == 'Dataset32') or (dir_ == 'Dataset33')):
        seq_type = 'brain'
    elif ((dir_ == 'Dataset20') or (dir_ == 'Dataset21')
          or (dir_ == 'Dataset22') or (dir_ == 'Dataset23')):
        seq_type = 'slitlamp'
    elif ((dir_ == 'Dataset10') or (dir_ == 'Dataset11')
          or (dir_ == 'Dataset12') or (dir_ == 'Dataset13')):
        seq_type = 'cochlea'
    else:
        seq_type = 'Unknown'

    return seq_type


#types = ['tweezer', 'brain', 'slitlamp', 'cochlea']
#types = ['brain', 'tweezer', 'slitlamp', 'cochlea']
types = ['cochlea', 'brain', 'tweezer', 'slitlamp']

res_dirs = OrderedDict()
pred_dirs = OrderedDict()

res_dirs['Dataset00'] = '2018-06-01_17-37-00_exp'
res_dirs['Dataset01'] = '2017-12-06_17-56-51_exp'
res_dirs['Dataset02'] = '2018-09-27_18-05-07_exp'
res_dirs['Dataset03'] = '2017-12-07_02-50-27_exp'
pred_dirs['Dataset00'] = 'tweezer_unet_obj_0'
pred_dirs['Dataset01'] = 'tweezer_unet_obj_1'
pred_dirs['Dataset02'] = 'tweezer_unet_obj_2'
pred_dirs['Dataset03'] = 'tweezer_unet_obj_3'

res_dirs['Dataset30'] = '2017-12-06_13-37-18_exp'
res_dirs['Dataset31'] = '2017-12-06_14-40-17_exp'
res_dirs['Dataset32'] = '2017-12-06_19-41-02_exp'
res_dirs['Dataset33'] = '2017-12-07_15-32-41_exp'
pred_dirs['Dataset30'] = 'brain_unet_obj_0'
pred_dirs['Dataset31'] = 'brain_unet_obj_1'
pred_dirs['Dataset32'] = 'brain_unet_obj_2'
pred_dirs['Dataset33'] = 'brain_unet_obj_3'

res_dirs['Dataset20'] = '2017-12-06_13-36-56_exp'
res_dirs['Dataset21'] = '2017-12-06_17-07-27_exp'
res_dirs['Dataset22'] = '2017-12-06_18-46-06_exp'
res_dirs['Dataset23'] = '2017-12-06_19-48-47_exp'
pred_dirs['Dataset20'] = 'slitlamp_unet_obj_0'
pred_dirs['Dataset21'] = 'slitlamp_unet_obj_1'
pred_dirs['Dataset22'] = 'slitlamp_unet_obj_2'
pred_dirs['Dataset23'] = 'slitlamp_unet_obj_3'

res_dirs['Dataset10'] = '2017-12-06_14-26-17_exp'
res_dirs['Dataset11'] = '2017-12-06_16-16-27_exp'
res_dirs['Dataset12'] = '2017-12-06_17-28-07_exp'
res_dirs['Dataset13'] = '2017-12-06_20-50-10_exp'
pred_dirs['Dataset10'] = 'cochlea_unet_obj_0'
pred_dirs['Dataset11'] = 'cochlea_unet_obj_1'
pred_dirs['Dataset12'] = 'cochlea_unet_obj_2'
pred_dirs['Dataset13'] = 'cochlea_unet_obj_3'
