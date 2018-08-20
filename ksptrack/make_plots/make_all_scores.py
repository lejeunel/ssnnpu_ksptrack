from sklearn.metrics import (precision_recall_curve)
from skimage import (color, segmentation)
import os
import datetime
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labeling.utils import my_utils as utls
from labeling.utils import learning_dataset
from labeling.utils import csv_utils as csv
from labeling.exps import results_dirs as rd

"""
Calculate (F1, PR, RC) in all results dirs and write score csv file
"""

n_decs = 2
n_sets_per_type = 1

dfs = []
all_f1 = []
all_pr = []
all_rc = []

# Self-learning
for key in rd.types:

    f1s_pm = []
    f1s_ksp = []
    f1s_vilar = []
    f1s_g2s = []
    f1s_mic17 = []
    f1s_wtp = []

    prs_pm = []
    prs_ksp = []
    prs_vilar = []
    prs_g2s = []
    prs_mic17 = []
    prs_wtp = []

    rcs_pm = []
    rcs_ksp = []
    rcs_vilar = []
    rcs_g2s = []
    rcs_mic17 = []
    rcs_wtp = []

    # Get first gaze-set of every dataset
    for i in range(4):
        #My model
        file_ksp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][i][0],
                                'metrics.npz')

        if(os.path.exists(file_ksp)):
            print('Loading: ' + file_ksp)
            np_ksp = np.load(file_ksp)

            # PM
            num = np_ksp['pr_pm'] * np_ksp['rc_pm']
            denum = np_ksp['pr_pm'] + np_ksp['rc_pm']
            f1_pm = 2*(num)/(denum)
            max_f1_pm = np.max(f1_pm)
            pr_pm = np_ksp['pr_pm'][0, np.argmax(f1_pm)]
            rc_pm = np_ksp['rc_pm'][0, np.argmax(f1_pm)]
            f1s_pm.append(max_f1_pm)
            prs_pm.append(pr_pm)
            rcs_pm.append(rc_pm)


            # KSP
            num = np_ksp['pr_ksp'] * np_ksp['rc_ksp']
            denum = np_ksp['pr_ksp'] + np_ksp['rc_ksp']
            f1_ksp = 2*(num)/(denum)
            max_f1_ksp = np.max(f1_ksp)
            pr_ksp = np_ksp['pr_ksp'][np.argmax(f1_ksp)]
            rc_ksp = np_ksp['rc_ksp'][np.argmax(f1_ksp)]
            f1s_ksp.append(max_f1_ksp)
            prs_ksp.append(pr_ksp)
            rcs_ksp.append(rc_ksp)
        else:
            print(file_ksp + ' does not exist')

        # What's the point
        import pdb; pdb.set_trace()
        file_wtp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_wtp[key][i],
                                'scores.csv')

        if(os.path.exists(file_wtp)):
            print('Loading: ' + file_wtp)
            df = pd.read_csv(file_wtp)
            f1_wtp = df.iloc[0]['F1']
            f1s_wtp.append(f1_wtp)
        else:
            print(file_wtp + ' does not exist')

        # Vilarino model
        file_vilar = os.path.join(rd.root_dir,
                                  rd.res_dirs_dict_vilar[key][i],
                                  'score.csv')
        if(os.path.exists(file_vilar)):
            print('Loading: ' + file_vilar)
            df_vilar = pd.read_csv(file_vilar)
            f1_vilar = df_vilar.loc[0][-1]
            f1s_vilar.append(f1_vilar)
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
        else:
            print(file_mic17 + ' does not exist')
            f1s_mic17.append(-1)

    all_f1.append(np.concatenate((np.asarray(f1s_pm).reshape(1,-1),
                                  np.mean(f1s_pm).reshape(1,1),
                                  np.std(f1s_pm).reshape(1,1)),axis=1))
    all_f1.append(np.concatenate((np.asarray(f1s_ksp).reshape(1,-1),
                                  np.mean(f1s_ksp).reshape(1,1),
                                  np.std(f1s_ksp).reshape(1,1)),axis=1))
    all_f1.append(np.concatenate((np.asarray(f1s_vilar).reshape(1,-1),
                                  np.mean(f1s_vilar).reshape(1,1),
                                  np.std(f1s_vilar).reshape(1,1)),axis=1))
    all_f1.append(np.concatenate((np.asarray(f1s_g2s).reshape(1,-1),
                                  np.mean(f1s_g2s).reshape(1,1),
                                  np.std(f1s_g2s).reshape(1,1)),axis=1))
    all_f1.append(np.concatenate((np.asarray(f1s_mic17).reshape(1,-1),
                                  np.mean(f1s_mic17).reshape(1,1),
                                  np.std(f1s_mic17).reshape(1,1)),axis=1))
    all_f1.append(np.concatenate((np.asarray(f1s_wtp).reshape(1,-1),
                                  np.mean(f1s_wtp).reshape(1,1),
                                  np.std(f1s_wtp).reshape(1,1)),axis=1))

C = pd.Index(['0', '1', '2', '3', 'All mean F1', 'All std F1'], name="columns")
indices = [rd.types, ['KSPopt', 'KSP', 'vilar', 'gaze2', 'mic17', 'wtp']]

I = pd.MultiIndex.from_product(indices, names=['Types', 'Methods'])

data = np.concatenate(all_f1)
data = np.round_(data, decimals = n_decs)
df = pd.DataFrame(data=data, index=I, columns=C)
dfs.append(df)
all_df = pd.concat(dfs)

prob_based_methods = ['mic17', 'wtp', 'KSPopt']
bin_based_methods = ['KSP', 'vilar', 'gaze2']
all_methods = prob_based_methods + bin_based_methods

for t in rd.types:
    file_out = os.path.join(rd.root_dir, 'plots_results', t + '_self.csv')
    print('Writing: ' + file_out)
    df = all_df.loc[t]

    # Bold on average on all sequences
    max_score = np.argmax(df.iloc[:,4][all_methods])
    others = [m for m in all_methods if(m != max_score)]
    df.loc[max_score, 'All do_bold'] = True
    df.loc[others, 'All do_bold'] = False

    # Bold per sequence
    for s in range(4):
        max_score = np.argmax(df.iloc[:,s][all_methods])
        others = [m for m in all_methods if(m != max_score)]
        df.loc[max_score, str(s) + '_do_bold'] = True
        df.loc[others, str(s) + '_do_bold'] = False

    df = df.reindex_axis(sorted(df.columns), axis=1)

    df.to_csv(path_or_buf=file_out, header=False)
