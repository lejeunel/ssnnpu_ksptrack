from sklearn.metrics import (precision_recall_curve)
from skimage import (color, segmentation)
import os
import datetime
import yaml
from labeling.utils import my_utils as utls
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labeling.utils import learning_dataset
from labeling.utils import csv_utils as csv
from labeling.exps import results_dirs as rd

"""
Reads (max) F1 scores in directories and make csv table
"""

n_decs = 2
n_sets_per_type = 1

dfs = []
all_scores = []

# Self-learning
for key in rd.types:

    f1s_pm = []
    f1s_ksp = []
    f1s_crf = []
    f1s_vilar = []
    f1s_g2s = []
    f1s_mic17 = []
    f1s_wtp = []

    prs_pm = []
    prs_ksp = []
    prs_crf = []
    prs_vilar = []
    prs_g2s = []
    prs_mic17 = []
    prs_wtp = []

    rcs_pm = []
    rcs_ksp = []
    rcs_crf = []
    rcs_vilar = []
    rcs_g2s = []
    rcs_mic17 = []
    rcs_wtp = []

    # Get first gaze-set of every dataset
    for i in range(4):
        #My model
        file_ksp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][i][0],
                                'scores.csv')

        if(os.path.exists(file_ksp)):
            print('Loading: ' + file_ksp)
            df = pd.read_csv(file_ksp)
            f1_pm = df.iloc[1]['F1']
            f1_ksp = df.iloc[0]['F1']
            pr_pm = df.iloc[1]['PR']
            pr_ksp = df.iloc[0]['PR']
            rc_pm = df.iloc[1]['RC']
            rc_ksp = df.iloc[0]['RC']
            #f1_crf = df.iloc[3]['F1']
            f1s_pm.append(f1_pm)
            f1s_ksp.append(f1_ksp)
            rcs_pm.append(rc_pm)
            rcs_ksp.append(rc_ksp)
            prs_pm.append(pr_pm)
            prs_ksp.append(pr_ksp)
            #f1s_crf.append(f1_crf)
        else:
            print(file_ksp + ' does not exist')

        # What's the point
        file_wtp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_wtp[key][i],
                                'scores.csv')

        if(os.path.exists(file_wtp)):
            print('Loading: ' + file_wtp)
            df = pd.read_csv(file_wtp)
            f1_wtp = df.iloc[0]['F1']
            f1s_wtp.append(f1_wtp)
            pr_wtp = df.iloc[0]['PR']
            prs_wtp.append(pr_wtp)
            rc_wtp = df.iloc[0]['RC']
            rcs_wtp.append(rc_wtp)
        else:
            print(file_wtp + ' does not exist')

        # Vilarino model
        file_vilar = os.path.join(rd.root_dir,
                                  rd.res_dirs_dict_vilar[key][i],
                                  'scores.csv')
        if(os.path.exists(file_vilar)):
            print('Loading: ' + file_vilar)
            df_vilar = pd.read_csv(file_vilar)
            f1_vilar = df_vilar.loc[0]['F1']
            f1s_vilar.append(f1_vilar)
            pr_vilar = df_vilar.loc[0]['PR']
            prs_vilar.append(pr_vilar)
            rc_vilar = df_vilar.loc[0]['RC']
            rcs_vilar.append(rc_vilar)
        else:
            print(file_vilar + ' does not exist.')
            f1s_vilar.append(0.15)

        # G2S model
        file_g2s = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_g2s[key][i],
                                'scores.csv')
        if(os.path.exists(file_g2s)):
            print('Loading: ' + file_g2s)
            df_g2s = pd.read_csv(file_g2s)
            f1_g2s = df_g2s.iloc[0]['F1']
            f1s_g2s.append(f1_g2s)
            pr_g2s = df_g2s.iloc[0]['PR']
            prs_g2s.append(pr_g2s)
            rc_g2s = df_g2s.iloc[0]['RC']
            rcs_g2s.append(rc_g2s)
        else:
            print(file_g2s + ' does not exist')
            f1s_g2s.append(-1)

        # MICCAI17 model
        file_mic17 = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_mic17[key][i],
                                'scores.csv')
        if(os.path.exists(file_mic17)):
            print('Loading: ' + file_mic17)
            df_mic17 = pd.read_csv(file_mic17)
            f1s_mic17.append(df_mic17.iloc[0]['F1'])
            prs_mic17.append(df_mic17.iloc[0]['PR'])
            rcs_mic17.append(df_mic17.iloc[0]['RC'])
        else:
            print(file_mic17 + ' does not exist')
            f1s_mic17.append(-1)

    all_scores.append(np.concatenate((np.asarray(f1s_pm).reshape(1,-1),
                                      np.asarray(prs_pm).reshape(1,-1),
                                      np.asarray(rcs_pm).reshape(1,-1),
                                      np.mean(f1s_pm).reshape(1,1),
                                      np.std(f1s_pm).reshape(1,1),
                                      np.mean(prs_pm).reshape(1,1),
                                      np.std(prs_pm).reshape(1,1),
                                      np.mean(rcs_pm).reshape(1,1),
                                      np.std(rcs_pm).reshape(1,1)),
                                     axis=1))
    all_scores.append(np.concatenate((np.asarray(f1s_ksp).reshape(1,-1),
                                      np.asarray(prs_ksp).reshape(1,-1),
                                      np.asarray(rcs_ksp).reshape(1,-1),
                                      np.mean(f1s_ksp).reshape(1,1),
                                      np.std(f1s_ksp).reshape(1,1),
                                      np.mean(prs_ksp).reshape(1,1),
                                      np.std(prs_ksp).reshape(1,1),
                                      np.mean(rcs_ksp).reshape(1,1),
                                      np.std(rcs_ksp).reshape(1,1)),
                                     axis=1))
    all_scores.append(np.concatenate((np.asarray(f1s_vilar).reshape(1,-1),
                                      np.asarray(prs_vilar).reshape(1,-1),
                                      np.asarray(rcs_vilar).reshape(1,-1),
                                      np.mean(f1s_vilar).reshape(1,1),
                                      np.std(f1s_vilar).reshape(1,1),
                                      np.mean(prs_vilar).reshape(1,1),
                                      np.std(prs_vilar).reshape(1,1),
                                      np.mean(rcs_vilar).reshape(1,1),
                                      np.std(rcs_vilar).reshape(1,1)),
                                     axis=1))
    all_scores.append(np.concatenate((np.asarray(f1s_g2s).reshape(1,-1),
                                      np.asarray(prs_g2s).reshape(1,-1),
                                      np.asarray(rcs_g2s).reshape(1,-1),
                                      np.mean(f1s_g2s).reshape(1,1),
                                      np.std(f1s_g2s).reshape(1,1),
                                      np.mean(prs_g2s).reshape(1,1),
                                      np.std(prs_g2s).reshape(1,1),
                                      np.mean(rcs_g2s).reshape(1,1),
                                      np.std(rcs_g2s).reshape(1,1)),
                                     axis=1))
    all_scores.append(np.concatenate((np.asarray(f1s_mic17).reshape(1,-1),
                                      np.asarray(prs_mic17).reshape(1,-1),
                                      np.asarray(rcs_mic17).reshape(1,-1),
                                      np.mean(f1s_mic17).reshape(1,1),
                                      np.std(f1s_mic17).reshape(1,1),
                                      np.mean(prs_mic17).reshape(1,1),
                                      np.std(prs_mic17).reshape(1,1),
                                      np.mean(rcs_mic17).reshape(1,1),
                                      np.std(rcs_mic17).reshape(1,1)),
                                     axis=1))
    all_scores.append(np.concatenate((np.asarray(f1s_wtp).reshape(1,-1),
                                      np.asarray(prs_wtp).reshape(1,-1),
                                      np.asarray(rcs_wtp).reshape(1,-1),
                                      np.mean(f1s_wtp).reshape(1,1),
                                      np.std(f1s_wtp).reshape(1,1),
                                      np.mean(prs_wtp).reshape(1,1),
                                      np.std(prs_wtp).reshape(1,1),
                                      np.mean(rcs_wtp).reshape(1,1),
                                      np.std(rcs_wtp).reshape(1,1)),
                                     axis=1))

C = pd.Index(['0_F1', '1_F1', '2_F1', '3_F1',
              '0_PR', '1_PR', '2_PR', '3_PR',
              '0_RC', '1_RC', '2_RC', '3_RC',
              'F1 mean', 'F1 std',
              'PR mean', 'PR std',
              'RC mean', 'RC std'
], name="columns")
indices = [rd.types, ['KSPopt', 'KSP', 'vilar', 'gaze2', 'mic17', 'wtp']]

I = pd.MultiIndex.from_product(indices, names=['Types', 'Methods'])

data = np.concatenate(all_scores)
data = np.round_(data, decimals = n_decs)
df = pd.DataFrame(data=data, index=I, columns=C)
dfs.append(df)
all_df = pd.concat(dfs)
all_df.sort_index(axis='columns', inplace=True)

file_out = os.path.join(rd.root_dir, 'plots_results', 'all_self.csv')
all_df.to_csv(path_or_buf=file_out, header=False)

prob_based_methods = ['mic17', 'wtp', 'KSPopt']
bin_based_methods = ['KSP', 'vilar', 'gaze2']
all_methods = prob_based_methods + bin_based_methods

for t in rd.types:
    file_out = os.path.join(rd.root_dir, 'plots_results', t + '_self.csv')
    print('Writing: ' + file_out)
    df = all_df.loc[t]

    # Bold on average on all sequences
    max_score = np.argmax(df['F1 mean'][all_methods])
    others = [m for m in all_methods if(m != max_score)]
    df.loc[max_score, 'All do_bold'] = True
    df.loc[others, 'All do_bold'] = False

    # Bold per sequence
    for s in range(4):
        max_score = np.argmax(df['{}_F1'.format(s)][all_methods])
        others = [m for m in all_methods if(m != max_score)]
        df.loc[max_score, str(s) + '_do_bold'] = True
        df.loc[others, str(s) + '_do_bold'] = False

    df = df.reindex_axis(sorted(df.columns), axis=1)

    df.to_csv(path_or_buf=file_out, header=False)
