from ruamel import yaml
import os
from collections import defaultdict
import copy

# Paths and frames/gaze priorities for plots
res_dirs_dict_g2s = dict()
res_dirs_dict_ksp_cov = dict()
res_dirs_dict_ksp_miss = dict()
res_dirs_dict_ksp_uni_bg = dict()
res_dirs_dict_ksp_uni_neigh = dict()
res_dirs_dict_ksp_cov_ref = dict()
res_dirs_dict_mic17 = dict()
res_dirs_dict_vilar = dict()
res_dirs_dict_wtp = dict()
res_dirs_dict_ksp = dict()
res_dirs_dict_ksp_rec = dict()
res_dirs_dict_ksp_overfeat = dict()
out_dirs_dict_ksp = dict()
learning_dirs_dict = dict()
all_frames_dict = dict()
self_frames_dict = dict()
iters_frames_dict = dict()
best_dict_ksp = dict() # What to plot priority (dataset, gaze-set)
best_folds_learning = dict()

root_dir = '/home/laurent.lejeune/medical-labeling/'

types = ['Tweezer', 'Brain', 'Slitlamp', 'Cochlea']

dirs_wtp = ['Dataset00/results/2017-12-06_15-46-22_exp',
            'Dataset01/results/2017-12-06_21-16-55_exp',
            'Dataset02/results/2017-12-07_01-09-20_exp',
            'Dataset03/results/2017-12-07_04-48-59_exp']

dirs_vilar = ['Dataset00/results/2017-10-03_08-44-26_exp_vilar',
              'Dataset01/results/2017-10-04_14-41-12_exp_vilar',
              'Dataset02/results/2017-10-27_15-14-21_exp_vilar',
              'Dataset03/results/2017-10-29_12-31-45_exp_vilar']

dirs_g2s = ['Dataset00/results/2017-10-02_16-12-55_exp_g2s',
            'Dataset01/results/2017-11-09_17-00-07_exp_g2s',
            'Dataset02/results/2017-11-09_17-28-26_exp_g2s',
            'Dataset03/results/2017-11-09_18-04-47_exp_g2s']

dirs_mic17 = ['Dataset00/results/F1Maps',
              'Dataset01/results/F1Maps',
              'Dataset02/results/F1Maps',
              'Dataset03/results/F1Maps']

dirs_ksp_cov = dict()
dirs_ksp_miss = dict()
dirs_ksp_noise_uni_bg = dict()
dirs_ksp_noise_neigh = dict()
dirs_ksp_cov_ref = dict()

dirs_ksp_rec = ['Dataset00/results/2017-12-06_13-35-55_exp',
                'Dataset01/results/2017-12-06_17-56-51_exp',
                'Dataset02/results/2017-12-06_22-37-01_exp',
                'Dataset03/results/2017-12-07_02-50-27_exp']

dirs_ksp_rec = [['Dataset00/results/2018-02-09_13-44-22_exp_unet_rec',
                 'Dataset00/results/2018-02-09_21-56-09_exp_unet_rec',
                 'Dataset00/results/2018-02-10_07-20-31_exp_unet_rec',
                 'Dataset00/results/2018-02-10_13-06-19_exp_unet_rec',
                 'Dataset00/results/2018-02-10_19-02-20_exp_unet_rec'],
                ['Dataset01/results/2018-02-11_01-46-10_exp_unet_rec',
                 'Dataset01/results/2018-02-11_08-29-27_exp_unet_rec',
                 'Dataset01/results/2018-02-11_15-30-05_exp_unet_rec',
                 'Dataset01/results/2018-02-11_21-52-24_exp_unet_rec',
                 'Dataset01/results/2018-02-12_03-47-41_exp_unet_rec'],
                ['Dataset02/results/2018-02-12_08-39-47_exp_unet_rec',
                 'Dataset02/results/2018-02-12_14-38-59_exp_unet_rec',
                 'Dataset02/results/2018-02-12_15-01-04_exp_unet_rec',
                 'Dataset02/results/2018-02-10_14-45-49_exp_unet_rec',
                 'Dataset02/results/2018-02-10_20-31-03_exp_unet_rec'],
                ['Dataset03/results/2018-02-11_10-57-48_exp_unet_rec',
                 'Dataset03/results/2018-02-11_10-57-48_exp_unet_rec',
                 'Dataset03/results/2018-02-11_10-57-48_exp_unet_rec',
                 'Dataset03/results/2018-02-11_04-41-55_exp_unet_rec',
                 'Dataset03/results/2018-02-11_10-57-48_exp_unet_rec']]

dirs_ksp_overfeat = [['Dataset00/results/2018-02-18_20-15-41_exp_overfeat',
                      'Dataset00/results/2018-02-19_01-26-55_exp_overfeat',
                      'Dataset00/results/2018-02-19_09-26-07_exp_overfeat',
                      'Dataset00/results/2018-02-19_12-17-38_exp_overfeat',
                      'Dataset00/results/2018-02-19_15-56-58_exp_overfeat'],
                     ['Dataset01/results/2018-02-19_09-26-51_exp_overfeat',
                      'Dataset01/results/2018-02-19_12-40-25_exp_overfeat',
                      'Dataset01/results/2018-02-19_15-33-06_exp_overfeat',
                      'Dataset01/results/2018-02-19_18-33-00_exp_overfeat',
                      'Dataset01/results/2018-02-19_18-33-00_exp_overfeat'],
                     ['Dataset02/results/2018-02-18_20-15-21_exp_overfeat',
                      'Dataset02/results/2018-02-19_00-30-21_exp_overfeat',
                      'Dataset02/results/2018-02-19_19-29-23_exp_overfeat',
                      'Dataset02/results/2018-02-19_22-55-27_exp_overfeat',
                      'Dataset02/results/2018-02-20_02-55-27_exp_overfeat'],
                     ['Dataset03/results/2018-02-20_02-55-30_exp_overfeat',
                      'Dataset03/results/2018-02-20_06-45-38_exp_overfeat',
                      'Dataset03/results/2018-02-20_10-12-40_exp_overfeat',
                      'Dataset03/results/2018-02-20_10-34-05_exp_overfeat',
                      'Dataset03/results/2018-02-20_09-34-27_exp_overfeat']]

dirs_ksp = [['Dataset00/results/2017-11-07_20-37-11_exp',
             'Dataset00/results/2017-11-07_12-11-22_exp',
             'Dataset00/results/2017-11-08_05-21-22_exp',
             'Dataset00/results/2017-11-08_14-31-28_exp',
             'Dataset00/results/2017-11-08_21-49-48_exp'],
            ['Dataset01/results/2017-11-07_12-11-32_exp',
             'Dataset01/results/2017-11-07_21-19-30_exp',
             'Dataset01/results/2017-11-08_05-09-03_exp',
             'Dataset01/results/2017-11-08_14-40-42_exp',
             'Dataset01/results/2017-11-08_22-10-39_exp'],
            ['Dataset02/results/2017-11-07_15-20-13_exp',
             'Dataset02/results/2017-11-08_00-51-22_exp',
             'Dataset02/results/2017-11-08_06-57-24_exp',
             'Dataset02/results/2017-11-08_17-11-00_exp',
             'Dataset02/results/2017-11-09_00-12-06_exp'],
            ['Dataset03/results/2017-11-07_08-14-29_exp',
             'Dataset03/results/2017-11-07_23-42-50_exp',
             'Dataset03/results/2017-11-08_21-23-26_exp',
             'Dataset03/results/2017-11-09_18-04-17_exp',
             'Dataset03/results/2017-11-10_01-32-19_exp']]
out_dirs = ['learning_multigaze_Tweezer_2017-11-10_11-00-30',
            'learning_multigaze_Tweezer_2017-11-10_11-34-33',
            'learning_multigaze_Tweezer_2017-11-10_12-09-16',
            'learning_multigaze_Tweezer_2017-11-10_12-43-33']
learning_dir = 'learning_Tweezer_2017-11-12_14-22-42'
all_frames_idx = [[10, 40, 80, 100],
              [10, 40, 80, 100],
              [10, 40, 80, 100],
              [10, 40, 80, 100]]
self_frames_idx = [40, 30, 40, 40]
iters_frames_idx = [57, 30, 40, 40]

res_dirs_dict_ksp['Tweezer'] = dirs_ksp
res_dirs_dict_ksp_rec['Tweezer'] = dirs_ksp_rec
res_dirs_dict_ksp_overfeat['Tweezer'] = dirs_ksp_overfeat
res_dirs_dict_vilar['Tweezer'] = dirs_vilar
res_dirs_dict_g2s['Tweezer'] = dirs_g2s
res_dirs_dict_mic17['Tweezer'] = dirs_mic17
res_dirs_dict_wtp['Tweezer'] = dirs_wtp
out_dirs_dict_ksp['Tweezer'] = out_dirs
learning_dirs_dict['Tweezer'] = learning_dir
all_frames_dict['Tweezer'] = all_frames_idx
self_frames_dict['Tweezer'] = self_frames_idx
iters_frames_dict['Tweezer'] = iters_frames_idx
best_dict_ksp['Tweezer'] = [(0,0), (1,0)]
best_folds_learning['Tweezer'] = [1,3,2,0]

dirs_vilar = ['Dataset30/results/2017-10-27_15-17-50_exp_vilar',
              'Dataset31/results/2017-11-02_11-36-54_exp_vilar',
              'Dataset32/results/2017-10-29_15-34-05_exp_vilar',
              'Dataset33/results/2017-11-03_09-06-38_exp_vilar']

dirs_g2s = ['Dataset30/results/2017-11-21_14-10-47_exp_g2s',
            'Dataset31/results/2017-11-21_14-24-57_exp_g2s',
            'Dataset32/results/2017-11-21_14-38-20_exp_g2s',
            'Dataset33/results/2017-11-21_14-51-51_exp_g2s']

dirs_mic17 = ['Dataset30/results/F1Maps',
              'Dataset31/results/F1Maps',
              'Dataset32/results/F1Maps',
              'Dataset33/results/F1Maps']

dirs_wtp = ['Dataset30/results/2017-12-06_15-48-22_exp',
            'Dataset31/results/2017-12-06_22-58-07_exp',
            'Dataset32/results/2017-12-07_04-34-35_exp',
            'Dataset33/results/2017-12-07_12-23-47_exp']

dirs_ksp_rec = ['Dataset30/results/2017-12-06_13-37-18_exp',
                'Dataset31/results/2017-12-06_14-40-17_exp',
                'Dataset32/results/2017-12-06_19-41-02_exp',
                'Dataset33/results/2017-11-09_04-19-06_exp']

dirs_ksp = [['Dataset30/results/2017-11-07_14-49-56_exp',
             'Dataset30/results/2017-11-07_17-50-30_exp',
             'Dataset30/results/2017-11-07_20-52-41_exp',
             'Dataset30/results/2017-11-08_00-23-08_exp',
             'Dataset30/results/2017-11-08_03-41-08_exp'],
            ['Dataset31/results/2017-11-09_14-43-11_exp',
             'Dataset31/results/2017-11-09_16-37-33_exp',
             'Dataset31/results/2017-11-09_19-20-03_exp',
             'Dataset31/results/2017-11-09_22-54-46_exp',
             'Dataset31/results/2017-11-10_02-17-26_exp'],
            ['Dataset32/results/2017-11-07_14-50-15_exp',
             'Dataset32/results/2017-11-07_22-33-41_exp',
             'Dataset32/results/2017-11-08_02-16-38_exp',
             'Dataset32/results/2017-11-08_07-12-57_exp',
             'Dataset32/results/2017-11-08_13-38-54_exp'],
            ['Dataset33/results/2017-11-08_10-07-34_exp',
             'Dataset33/results/2017-11-08_14-15-03_exp',
             'Dataset33/results/2017-11-08_21-35-29_exp',
             'Dataset33/results/2017-11-09_01-08-57_exp',
             'Dataset33/results/2017-11-09_04-19-06_exp']]

dirs_ksp_rec = [['Dataset30/results/2018-02-09_13-44-58_exp_unet_rec',
                 'Dataset30/results/2018-02-09_15-43-53_exp_unet_rec',
                 'Dataset30/results/2018-02-09_17-49-49_exp_unet_rec',
                 'Dataset30/results/2018-02-09_14-15-14_exp_unet_rec',
                 'Dataset30/results/2018-02-09_16-56-35_exp_unet_rec'],
                ['Dataset31/results/2018-02-10_00-19-08_exp_unet_rec',
                 'Dataset31/results/2018-02-10_01-55-59_exp_unet_rec',
                 'Dataset31/results/2018-02-10_04-14-19_exp_unet_rec',
                 'Dataset31/results/2018-02-09_19-17-38_exp_unet_rec',
                 'Dataset31/results/2018-02-09_21-31-42_exp_unet_rec'],
                ['Dataset32/results/2018-02-10_11-56-17_exp_unet_rec',
                 'Dataset32/results/2018-02-10_15-35-36_exp_unet_rec',
                 'Dataset32/results/2018-02-10_20-42-33_exp_unet_rec',
                 'Dataset32/results/2018-02-11_00-14-56_exp_unet_rec',
                 'Dataset32/results/2018-02-11_03-40-12_exp_unet_rec'],
                ['Dataset33/results/2018-02-11_07-55-28_exp_unet_rec',
                 'Dataset33/results/2018-02-11_10-00-43_exp_unet_rec',
                 'Dataset33/results/2018-02-11_13-02-25_exp_unet_rec',
                 'Dataset33/results/2018-02-11_15-30-08_exp_unet_rec',
                 'Dataset33/results/2018-02-11_19-30-17_exp_unet_rec']]

dirs_ksp_overfeat = [['Dataset30/results/2018-02-19_08-40-43_exp_overfeat',
                      'Dataset30/results/2018-02-19_09-05-31_exp_overfeat',
                      'Dataset30/results/2018-02-19_09-41-54_exp_overfeat',
                      'Dataset30/results/2018-02-19_10-16-24_exp_overfeat',
                      'Dataset30/results/2018-02-19_10-53-13_exp_overfeat'],
                     ['Dataset31/results/2018-02-19_11-27-45_exp_overfeat',
                      'Dataset31/results/2018-02-19_11-55-05_exp_overfeat',
                      'Dataset31/results/2018-02-19_12-22-07_exp_overfeat',
                      'Dataset31/results/2018-02-19_12-53-35_exp_overfeat',
                      'Dataset31/results/2018-02-19_13-24-46_exp_overfeat'],
                     ['Dataset32/results/2018-02-20_09-35-23_exp_overfeat',
                      'Dataset32/results/2018-02-20_11-12-54_exp_overfeat',
                      'Dataset32/results/2018-02-20_12-45-35_exp_overfeat',
                      'Dataset32/results/2018-02-20_14-26-43_exp_overfeat',
                      'Dataset32/results/2018-02-20_13-06-37_exp_overfeat'],
                     ['Dataset33/results/2018-02-20_09-35-32_exp_overfeat',
                      'Dataset33/results/2018-02-20_10-24-29_exp_overfeat',
                      'Dataset33/results/2018-02-20_11-48-14_exp_overfeat',
                      'Dataset33/results/2018-02-20_12-44-58_exp_overfeat',
                      'Dataset33/results/2018-02-20_13-44-05_exp_overfeat']]

out_dirs = ['learning_multigaze_Brain_2017-11-10_10-05-24',
            'learning_multigaze_Brain_2017-11-10_10-20-49',
            'learning_multigaze_Brain_2017-11-10_10-32-24',
            'learning_multigaze_Brain_2017-11-10_10-45-42']
learning_dir = 'learning_Brain_2017-11-12_14-22-35'
all_frames_idx = [[10, 20, 40, 60],
              [10, 20, 40, 60],
              [10, 20, 40, 60],
              [10, 20, 40, 60]]
self_frames_idx = [40, 30, 40, 40]
iters_frames_idx = [63, 30, 40, 40]

res_dirs_dict_ksp['Brain'] = dirs_ksp
res_dirs_dict_ksp_rec['Brain'] = dirs_ksp_rec
res_dirs_dict_ksp_overfeat['Brain'] = dirs_ksp_overfeat
res_dirs_dict_vilar['Brain'] = dirs_vilar
res_dirs_dict_mic17['Brain'] = dirs_mic17
res_dirs_dict_wtp['Brain'] = dirs_wtp
res_dirs_dict_g2s['Brain'] = dirs_g2s
out_dirs_dict_ksp['Brain'] = out_dirs
learning_dirs_dict['Brain'] = learning_dir
all_frames_dict['Brain'] = all_frames_idx
self_frames_dict['Brain'] = self_frames_idx
iters_frames_dict['Brain'] = iters_frames_idx
best_dict_ksp['Brain'] = [(0,0), (1,0)]
best_folds_learning['Brain'] = [0,1,2,3]

dirs_vilar = ['Dataset20/results/2017-10-27_15-17-35_exp_vilar',
              'Dataset21/results/2017-10-31_14-33-06_exp_vilar',
              'Dataset22/results/2017-11-02_09-28-37_exp_vilar',
              'Dataset23/results/2017-11-04_16-39-06_exp_vilar']

dirs_g2s = ['Dataset20/results/2017-11-21_14-12-49_exp_g2s',
            'Dataset21/results/2017-11-21_14-39-36_exp_g2s',
            'Dataset22/results/2017-11-21_15-10-07_exp_g2s',
            'Dataset23/results/2017-11-21_15-31-22_exp_g2s']

dirs_wtp = ['Dataset20/results/2017-12-06_15-47-49_exp',
            'Dataset21/results/2017-12-06_21-28-02_exp',
            'Dataset22/results/2017-12-07_02-44-35_exp',
            'Dataset23/results/2017-12-07_06-53-49_exp']

dirs_ksp_rec = ['Dataset20/results/2017-12-06_13-36-56_exp',
                'Dataset21/results/2017-12-06_17-07-27_exp',
                'Dataset22/results/2017-12-06_18-46-06_exp',
                'Dataset23/results/2017-12-06_19-48-47_exp']

dirs_mic17 = ['Dataset20/results/F1Maps',
              'Dataset21/results/F1Maps',
              'Dataset22/results/F1Maps',
              'Dataset23/results/F1Maps']

dirs_ksp = [['Dataset20/results/2017-11-06_17-01-54_exp',
             'Dataset20/results/2017-11-06_22-15-38_exp',
             'Dataset20/results/2017-11-07_02-51-07_exp',
             'Dataset20/results/2017-11-07_05-42-27_exp',
             'Dataset20/results/2017-11-07_09-18-04_exp'],
            ['Dataset21/results/2017-11-06_18-01-13_exp',
             'Dataset21/results/2017-11-06_22-09-23_exp',
             'Dataset21/results/2017-11-07_00-15-32_exp',
             'Dataset21/results/2017-11-07_01-38-18_exp',
             'Dataset21/results/2017-11-07_02-57-33_exp'],
            ['Dataset22/results/2017-11-07_08-08-47_exp',
             'Dataset22/results/2017-11-07_11-21-12_exp',
             'Dataset22/results/2017-11-07_14-10-08_exp',
             'Dataset22/results/2017-11-07_18-23-36_exp',
             'Dataset22/results/2017-11-09_09-55-17_exp'],
            ['Dataset23/results/2017-11-06_20-15-52_exp',
             'Dataset23/results/2017-11-07_10-42-13_exp',
             'Dataset23/results/2017-11-07_16-13-23_exp',
             'Dataset23/results/2017-11-07_23-03-54_exp',
             'Dataset23/results/2017-11-08_16-41-48_exp']]

dirs_ksp_rec = [['Dataset20/results/2018-02-09_13-44-51_exp_unet_rec',
                 'Dataset20/results/2018-02-09_16-50-06_exp_unet_rec',
                 'Dataset20/results/2018-02-09_20-01-52_exp_unet_rec',
                 'Dataset20/results/2018-02-09_14-15-50_exp_unet_rec',
                 'Dataset20/results/2018-02-10_01-28-44_exp_unet_rec'],
                ['Dataset21/results/2018-02-10_04-23-41_exp_unet_rec',
                 'Dataset21/results/2018-02-10_07-33-36_exp_unet_rec',
                 'Dataset21/results/2018-02-10_09-28-37_exp_unet_rec',
                 'Dataset21/results/2018-02-10_11-43-21_exp_unet_rec',
                 'Dataset21/results/2018-02-10_13-51-10_exp_unet_rec'],
                ['Dataset22/results/2018-02-10_15-32-57_exp_unet_rec',
                 'Dataset22/results/2018-02-10_16-48-17_exp_unet_rec',
                 'Dataset22/results/2018-02-10_18-06-30_exp_unet_rec',
                 'Dataset22/results/2018-02-10_19-45-22_exp_unet_rec',
                 'Dataset22/results/2018-02-10_21-13-30_exp_unet_rec'],
                ['Dataset23/results/2018-02-10_22-53-08_exp_unet_rec',
                 'Dataset23/results/2018-02-11_03-48-18_exp_unet_rec',
                 'Dataset23/results/2018-02-11_08-14-53_exp_unet_rec',
                 'Dataset23/results/2018-02-11_13-33-03_exp_unet_rec',
                 'Dataset23/results/2018-02-11_17-38-46_exp_unet_rec']]

dirs_ksp_overfeat = [['Dataset20/results/2018-02-19_08-31-12_exp_overfeat',
                      'Dataset20/results/2018-02-19_10-59-19_exp_overfeat',
                      'Dataset20/results/2018-02-19_13-32-50_exp_overfeat',
                      'Dataset20/results/2018-02-19_15-59-34_exp_overfeat',
                      'Dataset20/results/2018-02-19_18-14-26_exp_overfeat'],
                     ['Dataset21/results/2018-02-19_20-31-03_exp_overfeat',
                      'Dataset21/results/2018-02-19_21-53-26_exp_overfeat',
                      'Dataset21/results/2018-02-19_23-10-23_exp_overfeat',
                      'Dataset21/results/2018-02-20_00-32-38_exp_overfeat',
                      'Dataset21/results/2018-02-20_01-43-05_exp_overfeat'],
                     ['Dataset22/results/2018-02-19_08-32-24_exp_overfeat',
                      'Dataset22/results/2018-02-19_09-33-56_exp_overfeat',
                      'Dataset22/results/2018-02-19_10-35-54_exp_overfeat',
                      'Dataset22/results/2018-02-19_10-35-54_exp_overfeat',
                      'Dataset22/results/2018-02-19_12-42-04_exp_overfeat'],
                     ['Dataset23/results/2018-02-19_13-39-31_exp_overfeat',
                      'Dataset23/results/2018-02-19_16-25-23_exp_overfeat',
                      'Dataset23/results/2018-02-19_18-54-14_exp_overfeat',
                      'Dataset23/results/2018-02-19_20-54-00_exp_overfeat',
                      'Dataset23/results/2018-02-19_23-04-05_exp_overfeat']]

out_dirs = ['learning_multigaze_Slitlamp_2017-11-10_14-52-57',
            'learning_multigaze_Slitlamp_2017-11-10_15-25-24',
            'learning_multigaze_Slitlamp_2017-11-10_15-54-30',
            'learning_multigaze_Slitlamp_2017-11-10_16-07-38']
learning_dir = 'learning_Slitlamp_2017-11-24_10-29-15'

all_frames_idx = [[10, 20, 40, 60],
              [10, 20, 40, 60],
              [10, 20, 40, 60],
              [10, 20, 40, 60]]
self_frames_idx = [17, 30, 40, 40]
iters_frames_idx = [9, 30, 40, 40]

res_dirs_dict_ksp['Slitlamp'] = dirs_ksp
res_dirs_dict_ksp_rec['Slitlamp'] = dirs_ksp_rec
res_dirs_dict_ksp_overfeat['Slitlamp'] = dirs_ksp_overfeat
res_dirs_dict_vilar['Slitlamp'] = dirs_vilar
res_dirs_dict_mic17['Slitlamp'] = dirs_mic17
res_dirs_dict_wtp['Slitlamp'] = dirs_wtp
res_dirs_dict_g2s['Slitlamp'] = dirs_g2s
out_dirs_dict_ksp['Slitlamp'] = out_dirs
learning_dirs_dict['Slitlamp'] = learning_dir
all_frames_dict['Slitlamp'] = all_frames_idx
self_frames_dict['Slitlamp'] = self_frames_idx
iters_frames_dict['Slitlamp'] = iters_frames_idx
best_dict_ksp['Slitlamp'] = [0, 1]
best_dict_ksp['Slitlamp'] = [(0,0), (1,0)]
best_folds_learning['Slitlamp'] = [0,3,2,1]

dirs_vilar = ['Dataset10/results/2017-10-27_15-16-57_exp_vilar',
              'Dataset11/results/2017-10-29_02-56-42_exp_vilar',
              'Dataset12/results/2018-02-19_10-46-36_exp_vilar',
              'Dataset13/results/2017-10-31_14-06-19_exp_vilar']

dirs_g2s = ['Dataset10/results/2017-11-21_17-15-00_exp_g2s',
            'Dataset11/results/2017-11-21_17-28-59_exp_g2s',
            'Dataset12/results/2017-11-21_17-44-49_exp_g2s',
            'Dataset13/results/2017-11-21_18-13-45_exp_g2s']

dirs_wtp = ['Dataset10/results/2017-12-06_15-46-46_exp',
            'Dataset11/results/2017-12-06_20-01-46_exp',
            'Dataset12/results/2017-12-06_22-03-30_exp',
            'Dataset13/results/2017-12-07_01-39-05_exp']

dirs_ksp_rec = ['Dataset10/results/2017-12-06_14-26-17_exp',
                'Dataset11/results/2017-12-06_16-16-27_exp',
                'Dataset12/results/2017-12-06_17-28-07_exp',
                'Dataset13/results/2017-12-06_20-50-10_exp']

dirs_mic17 = ['Dataset10/results/F1Maps',
              'Dataset11/results/F1Maps',
              'Dataset12/results/F1Maps',
              'Dataset13/results/F1Maps']

dirs_ksp = [['Dataset10/results/2017-11-07_04-44-16_exp',
             'Dataset10/results/2017-11-06_17-04-13_exp',
             'Dataset10/results/2017-11-06_19-49-44_exp',
             'Dataset10/results/2017-11-06_23-09-46_exp',
             'Dataset10/results/2017-11-07_01-35-53_exp'],
            ['Dataset11/results/2017-11-06_20-42-54_exp',
             'Dataset11/results/2017-11-03_19-08-16_exp',
             'Dataset11/results/2017-11-06_18-15-30_exp',
             'Dataset11/results/2017-11-06_19-33-03_exp',
             'Dataset11/results/2017-11-06_22-04-22_exp'],
            ['Dataset12/results/2017-11-06_18-04-58_exp',
             'Dataset12/results/2017-11-07_00-16-26_exp',
             'Dataset12/results/2017-11-07_06-27-23_exp',
             'Dataset12/results/2017-11-07_14-50-13_exp',
             'Dataset12/results/2017-11-07_23-36-48_exp'],
            ['Dataset13/results/2017-11-07_08-11-12_exp',
             'Dataset13/results/2017-11-07_12-05-18_exp',
             'Dataset13/results/2017-11-07_22-03-55_exp',
             'Dataset13/results/2017-11-08_15-19-06_exp',
             'Dataset13/results/2017-11-08_21-51-36_exp']]

dirs_ksp_rec = [['Dataset10/results/2018-02-09_13-44-42_exp_unet_rec',
                 'Dataset10/results/2018-02-09_16-48-31_exp_unet_rec',
                 'Dataset10/results/2018-02-09_19-56-19_exp_unet_rec',
                 'Dataset10/results/2018-02-09_14-07-46_exp_unet_rec',
                 'Dataset10/results/2018-02-09_16-55-49_exp_unet_rec'],
                ['Dataset11/results/2018-02-10_05-25-54_exp_unet_rec',
                 'Dataset11/results/2018-02-10_06-38-58_exp_unet_rec',
                 'Dataset11/results/2018-02-10_07-57-57_exp_unet_rec',
                 'Dataset11/results/2018-02-10_09-01-53_exp_unet_rec',
                 'Dataset11/results/2018-02-10_10-08-56_exp_unet_rec'],
                ['Dataset12/results/2018-02-10_11-15-44_exp_unet_rec',
                 'Dataset12/results/2018-02-10_15-13-28_exp_unet_rec',
                 'Dataset12/results/2018-02-10_20-01-10_exp_unet_rec',
                 'Dataset12/results/2018-02-11_01-04-59_exp_unet_rec',
                 'Dataset12/results/2018-02-11_05-58-51_exp_unet_rec'],
                ['Dataset13/results/2018-02-11_09-43-39_exp_unet_rec',
                 'Dataset13/results/2018-02-11_12-20-17_exp_unet_rec',
                 'Dataset13/results/2018-02-11_15-23-03_exp_unet_rec',
                 'Dataset13/results/2018-02-11_18-20-13_exp_unet_rec',
                 'Dataset13/results/2018-02-11_21-07-41_exp_unet_rec']]

dirs_ksp_overfeat = [['Dataset10/results/2018-02-19_08-29-24_exp_overfeat',
                      'Dataset10/results/2018-02-19_09-54-35_exp_overfeat',
                      'Dataset10/results/2018-02-19_11-16-25_exp_overfeat',
                      'Dataset10/results/2018-02-19_12-42-59_exp_overfeat',
                      'Dataset10/results/2018-02-19_14-09-00_exp_overfeat'],
                     ['Dataset11/results/2018-02-19_15-33-01_exp_overfeat',
                      'Dataset11/results/2018-02-19_16-14-34_exp_overfeat',
                      'Dataset11/results/2018-02-19_17-11-23_exp_overfeat',
                      'Dataset11/results/2018-02-19_17-54-38_exp_overfeat',
                      'Dataset11/results/2018-02-19_18-35-17_exp_overfeat'],
                     ['Dataset12/results/2018-02-19_08-30-15_exp_overfeat',
                      'Dataset12/results/2018-02-19_10-39-15_exp_overfeat',
                      'Dataset12/results/2018-02-19_12-54-44_exp_overfeat',
                      'Dataset12/results/2018-02-19_14-55-53_exp_overfeat',
                      'Dataset12/results/2018-02-19_17-13-51_exp_overfeat'],
                     ['Dataset13/results/2018-02-19_19-09-39_exp_overfeat',
                      'Dataset13/results/2018-02-19_20-31-15_exp_overfeat',
                      'Dataset13/results/2018-02-19_21-55-27_exp_overfeat',
                      'Dataset13/results/2018-02-19_23-12-17_exp_overfeat',
                      'Dataset13/results/2018-02-20_00-37-07_exp_overfeat']]

out_dirs = ['learning_multigaze_Cochlea_2017-11-10_13-16-14',
            'learning_multigaze_Cochlea_2017-11-10_13-39-00',
            'learning_multigaze_Cochlea_2017-11-10_13-59-52',
            'learning_multigaze_Cochlea_2017-11-10_14-29-05']
learning_dir = 'learning_Cochlea_2017-11-23_16-28-08'

all_frames_idx = [[10, 20, 40, 60],
              [10, 20, 40, 60],
              [10, 20, 40, 60],
              [10, 20, 40, 60]]
self_frames_idx = [40, 30, 40, 40]
iters_frames_idx = [47, 30, 40, 40]

res_dirs_dict_ksp['Cochlea'] = dirs_ksp
res_dirs_dict_ksp_rec['Cochlea'] = dirs_ksp_rec
res_dirs_dict_ksp_overfeat['Cochlea'] = dirs_ksp_overfeat
res_dirs_dict_vilar['Cochlea'] = dirs_vilar
res_dirs_dict_mic17['Cochlea'] = dirs_mic17
res_dirs_dict_g2s['Cochlea'] = dirs_g2s
res_dirs_dict_wtp['Cochlea'] = dirs_wtp
out_dirs_dict_ksp['Cochlea'] = out_dirs
learning_dirs_dict['Cochlea'] = learning_dir
all_frames_dict['Cochlea'] = all_frames_idx
self_frames_dict['Cochlea'] = self_frames_idx
iters_frames_dict['Cochlea'] = iters_frames_idx
best_dict_ksp['Cochlea'] = [0, 1]
best_dict_ksp['Cochlea'] = [(0,0), (1,0)]
best_folds_learning['Cochlea'] = [0,2,1,3]

res_dirs_dict_ksp_miss['Dataset00'] = dict()
dirs_ksp_miss[0] = '2017-11-07_20-37-11_exp'
dirs_ksp_miss[5] = '2018-06-01_16-19-33_exp'
dirs_ksp_miss[10] = '2018-06-01_14-24-05_exp'
dirs_ksp_miss[20] = '2018-06-01_14-24-14_exp'
dirs_ksp_miss[40] = '2018-06-01_14-51-54_exp'
dirs_ksp_miss[50] = '2018-06-01_12-24-40_exp'
res_dirs_dict_ksp_miss['Dataset00'] = copy.deepcopy(dirs_ksp_miss)

res_dirs_dict_ksp_uni_neigh['Dataset00'] = dict()
dirs_ksp_noise_uni_bg = dict()
dirs_ksp_noise_uni_bg[0] = '2017-11-07_20-37-11_exp'
dirs_ksp_noise_uni_bg[5] = '2018-05-31_14-14-35_exp'
dirs_ksp_noise_uni_bg[10] = '2018-05-31_09-55-01_exp'
dirs_ksp_noise_uni_bg[20] = '2018-05-31_14-14-35_exp'
dirs_ksp_noise_uni_bg[40] = '2018-05-31_14-14-35_exp'
dirs_ksp_noise_uni_bg[50] = '2018-05-31_13-39-26_exp'
res_dirs_dict_ksp_uni_bg['Dataset00'] = copy.deepcopy(dirs_ksp_noise_uni_bg)

res_dirs_dict_ksp_uni_neigh['Dataset00'] = dict()
dist = 5
dirs_ksp_noise_neigh = dict()
dirs_ksp_noise_neigh[0] = '2017-11-07_20-37-11_exp'
dirs_ksp_noise_neigh[5] = '2018-06-03_15-50-30_exp'
dirs_ksp_noise_neigh[10] = '2018-06-03_12-43-28_exp'
dirs_ksp_noise_neigh[20] = '2018-06-01_09-23-47_exp'
dirs_ksp_noise_neigh[40] = '2018-06-01_16-19-46_exp'
dirs_ksp_noise_neigh[50] = '2018-05-31_09-57-08_exp'
res_dirs_dict_ksp_uni_neigh['Dataset00'][dist] = dict()
res_dirs_dict_ksp_uni_neigh['Dataset00'][dist] = copy.deepcopy(dirs_ksp_noise_neigh)

dist = 10
dirs_ksp_noise_neigh = dict()
dirs_ksp_noise_neigh[0] = '2017-11-07_20-37-11_exp'
dirs_ksp_noise_neigh[5] = '2018-06-01_17-53-25_exp'
dirs_ksp_noise_neigh[10] = '2018-05-31_09-55-40_exp'
dirs_ksp_noise_neigh[20] = '2018-06-01_09-23-47_exp'
dirs_ksp_noise_neigh[40] = '2018-05-31_09-56-40_exp'
dirs_ksp_noise_neigh[50] = '2018-06-03_14-14-09_exp'
res_dirs_dict_ksp_uni_neigh['Dataset00'][dist] = dict()
res_dirs_dict_ksp_uni_neigh['Dataset00'][dist] = copy.deepcopy(dirs_ksp_noise_neigh)

dirs_ksp_cov[20] = 'Dataset00/results/2017-12-02_19-42-58_exp'
dirs_ksp_cov[40] = 'Dataset00/results/2017-12-02_13-30-39_exp'
dirs_ksp_cov[60] = 'Dataset00/results/2017-12-02_20-25-16_exp'
dirs_ksp_cov[75] = 'Dataset00/results/2017-12-02_13-29-04_exp'
dirs_ksp_cov[90] = 'Dataset00/results/2017-11-28_10-22-31_exp'
res_dirs_dict_ksp_cov['Dataset00'] = copy.deepcopy(dirs_ksp_cov)

dirs_ksp_cov_ref[20] = 'Dataset00/results/2017-11-27_11-53-28_coverage'
dirs_ksp_cov_ref[40] = 'Dataset00/results/2017-11-27_11-45-53_coverage'
dirs_ksp_cov_ref[60] = 'Dataset00/results/2017-11-27_11-33-13_coverage'
dirs_ksp_cov_ref[75] = 'Dataset00/results/2017-11-27_11-25-46_coverage'
dirs_ksp_cov_ref[90] = 'Dataset00/results/2017-11-27_11-13-11_coverage'
res_dirs_dict_ksp_cov_ref['Dataset00'] = copy.deepcopy(dirs_ksp_cov_ref)

dirs_ksp_cov[20] = 'Dataset01/results/2017-12-02_19-34-30_exp'
dirs_ksp_cov[40] = 'Dataset01/results/2017-12-02_13-34-01_exp'
dirs_ksp_cov[60] = 'Dataset01/results/2017-12-02_20-50-43_exp'
dirs_ksp_cov[75] = 'Dataset01/results/2017-12-02_13-31-47_exp'
dirs_ksp_cov[90] = 'Dataset01/results/2017-11-28_10-22-40_exp'
res_dirs_dict_ksp_cov['Dataset01'] = copy.deepcopy(dirs_ksp_cov)

dirs_ksp_cov[20] = 'Dataset02/results/2017-12-02_20-29-27_exp'
dirs_ksp_cov[40] = 'Dataset02/results/2017-12-02_13-36-07_exp'
dirs_ksp_cov[60] = 'Dataset02/results/2017-12-02_20-32-02_exp'
dirs_ksp_cov[75] = 'Dataset02/results/2017-12-02_13-35-07_exp'
dirs_ksp_cov[90] = 'Dataset02/results/2017-11-28_10-22-45_exp'
res_dirs_dict_ksp_cov['Dataset02'] = copy.deepcopy(dirs_ksp_cov)

dirs_ksp_cov[20] = 'Dataset03/results/2017-12-02_18-59-00_exp'
dirs_ksp_cov[40] = 'Dataset03/results/2017-12-02_13-38-17_exp'
dirs_ksp_cov[60] = None
dirs_ksp_cov[75] = 'Dataset03/results/2017-12-02_13-37-29_exp'
dirs_ksp_cov[90] = 'Dataset03/results/2017-11-28_10-22-50_exp'
res_dirs_dict_ksp_cov['Dataset03'] = copy.deepcopy(dirs_ksp_cov)


# Initialize confs_dict
nested_dict = lambda: defaultdict(nested_dict)
confs_dict_ksp = nested_dict()

from ruamel.yaml import YAML
from ksptrack.cfgs import cfg

yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)

for key in res_dirs_dict_ksp.keys():
    for dset in range(len(res_dirs_dict_ksp[key])):
        for gset in range(len(res_dirs_dict_ksp[key][dset])):
            path_ = os.path.join(root_dir,
                                 res_dirs_dict_ksp[key][dset][gset],
                                 'cfg.yml')

            confs_dict_ksp[key][dset][gset] = cfg.load_and_convert(path_)
