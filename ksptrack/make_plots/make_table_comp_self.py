from sklearn.metrics import (precision_recall_curve)
from skimage import (color, segmentation)
import os
import datetime
import yaml
import my_utils as utls
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import learning_dataset
import gazeCsv as gaze
import results_dirs as rd

"""
Reads (max) F1 scores in directories and make csv table
"""

n_decs = 3
n_sets_per_type = 1

dfs = []
all_f1 = []

# Self-learning
for key in rd.types:

    f1s_ksp = []
    f1s_ksp_rec = []

    # Get first gaze-set of every dataset
    for i in range(4):
        #My model
        file_ksp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][i][0],
                                'scores.csv')

        file_ksp_rec = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp_rec[key][i],
                                'scores.csv')

        if(os.path.exists(file_ksp)):
            print('Loading: ' + file_ksp)
            df = pd.read_csv(file_ksp)
            f1_ksp = df.iloc[0]['F1']
            f1s_ksp.append(f1_ksp)
        else:
            print(file_ksp + ' does not exist')

        if(os.path.exists(file_ksp_rec)):
            print('Loading: ' + file_ksp_rec)
            df = pd.read_csv(file_ksp_rec)
            f1_ksp_rec = df.iloc[0]['F1']
            f1s_ksp_rec.append(f1_ksp_rec)
        else:
            print(file_ksp_rec + ' does not exist')


    all_f1.append(np.concatenate((np.asarray(f1s_ksp).reshape(1,-1),
                                  np.mean(f1s_ksp).reshape(1,1),
                                  np.std(f1s_ksp).reshape(1,1)),axis=1))
    all_f1.append(np.concatenate((np.asarray(f1s_ksp_rec).reshape(1,-1),
                                  np.mean(f1s_ksp_rec).reshape(1,1),
                                  np.std(f1s_ksp_rec).reshape(1,1)),axis=1))

C = pd.Index(['F1', 'F1','F1','F1','mean F1', 'std F1', 'do_bold'], name="columns")
indices = [rd.types, ['KSP (gaze-based)', 'KSP (U-Net)']]

I = pd.MultiIndex.from_product(indices, names=['Types', 'Methods'])

data = np.concatenate(all_f1)
data = np.round_(data, decimals = n_decs)
df = pd.DataFrame(data=data, index=I)
dfs.append(df)
all_df = pd.concat(dfs)

for t in rd.types:
    file_out = os.path.join(rd.root_dir, 'plots_results', t + '_self_feat_comp.csv')
    print('Writing: ' + file_out)
    df = all_df.loc[t]
    max_score_method = np.argmax(df.iloc[:,4])
    others = [id_ for id_ in df.index if(id_ != max_score_method)]
    df.loc[max_score_method,'do_bold'] = True
    df.loc[others,'do_bold'] = False
    df.to_csv(path_or_buf=file_out, header=False)
