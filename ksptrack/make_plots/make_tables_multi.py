from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib
import my_utils as utls
import plot_results_ksp_simple_multigaze as pksp
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import learning_dataset as ld
from skimage import (color, segmentation, util, transform)
import gazeCsv as gaze
import results_dirs as rd
import shutil as sh

n_decs = 2
dsets_per_type = 1

out_result_dir = os.path.join(rd.root_dir, 'plots_results')


# Make Inter-viewer table / Get frames

all_f1_ksp = []
all_f1_ksp_rec = []
all_f1_ksp_overfeat = []

for key in rd.types:
    f1 = []
    f1_rec = []
    f1_overfeat = []

    # KSP...
    for dset in range(len(rd.res_dirs_dict_ksp[key])):
        for g in range(len(rd.res_dirs_dict_ksp[key][dset])):
            print('Loading: ' + str(rd.res_dirs_dict_ksp[key][dset][g]))
            path_ = os.path.join(rd.root_dir,
                                    rd.res_dirs_dict_ksp[key][dset][g])
            df = pd.read_csv(os.path.join(path_,'scores.csv'))
            f1.append(np.round_(np.asarray((df.loc[0][-1])),
                                decimals = n_decs))

    f1_ksp = np.asarray(np.split(np.asarray(f1), 4))
    f1_ksp = np.asarray([(np.mean(f1_ksp[i,:]), np.std(f1_ksp[i,:])) for i in range(f1_ksp.shape[0])])
    f1_ksp = np.round_(f1_ksp.flatten(), decimals = n_decs)

    all_f1_ksp.append(f1_ksp)

    # KSP without prior objectness
    for dset in range(len(rd.res_dirs_dict_ksp_rec[key])):
        for g in range(len(rd.res_dirs_dict_ksp_rec[key][dset])):
            print('Loading: ' + str(rd.res_dirs_dict_ksp_rec[key][dset][g]))
            path_ = os.path.join(rd.root_dir,
                                    rd.res_dirs_dict_ksp_rec[key][dset][g])
            df = pd.read_csv(os.path.join(path_,'scores.csv'))
            f1_rec.append(np.round_(np.asarray(df.loc[0][1]), decimals = n_decs))

    f1_rec = np.asarray(np.split(np.asarray(f1_rec), 4))
    f1_rec = np.asarray([(np.mean(f1_rec[i,:]), np.std(f1_rec[i,:])) for i in range(f1_rec.shape[0])])
    f1_rec = np.round_(f1_rec.flatten(), decimals = n_decs)

    all_f1_ksp_rec.append(f1_rec)

    # KSP with overfeat features
    for dset in range(len(rd.res_dirs_dict_ksp_overfeat[key])):
        for g in range(len(rd.res_dirs_dict_ksp_overfeat[key][dset])):
            print('Loading: ' + str(rd.res_dirs_dict_ksp_overfeat[key][dset][g]))
            path_ = os.path.join(rd.root_dir,
                                 rd.res_dirs_dict_ksp_overfeat[key][dset][g])
            df = pd.read_csv(os.path.join(path_, 'scores.csv'))
            f1_overfeat.append(np.round_(np.asarray(df.loc[0][1]), decimals = n_decs))

    f1_overfeat = np.asarray(np.split(np.asarray(f1_overfeat), 4))
    f1_overfeat = np.asarray([(np.mean(f1_overfeat[i,:]), np.std(f1_overfeat[i,:])) for i in range(f1_overfeat.shape[0])])
    f1_overfeat = np.round_(f1_overfeat.flatten(), decimals = n_decs)

    all_f1_ksp_overfeat.append(f1_overfeat)

# Write table with F1 scores
file_out = os.path.join(out_result_dir,
                        'table_multigaze.csv')
C = pd.Index(['F1', 'std F1', 'F1', 'std F1', 'F1', 'std F1', 'F1', 'std F1', 'all means', 'all_std'], name="columns")
I = pd.Index(rd.types, name="Types")

data_F1 = np.asarray(all_f1_ksp)
F1_means_all = np.mean(data_F1[:,[0,2,4,6]], axis=1).reshape(-1,1)
F1_std_all = np.mean(data_F1[:,[1,3,5,7]], axis=1).reshape(-1,1)
data_F1_op = np.hstack((data_F1, F1_means_all, F1_std_all ))
print('Writing: ' + file_out)
df_ksp = pd.DataFrame(data=data_F1_op, index=I, columns=C)
df_ksp.to_csv(path_or_buf=file_out)

# Write table with F1 scores
file_out = os.path.join(out_result_dir,
                        'table_multigaze_rec.csv')
data_F1 = np.asarray(all_f1_ksp_rec)
F1_means_all = np.mean(data_F1[:,[0,2,4,6]], axis=1).reshape(-1,1)
F1_std_all = np.mean(data_F1[:,[1,3,5,7]], axis=1).reshape(-1,1)
data_F1_L2 = np.hstack((data_F1, F1_means_all, F1_std_all ))
print('Writing: ' + file_out)
df_ksp_rec = pd.DataFrame(data=data_F1_L2, index=I, columns=C)
df_ksp_rec.to_csv(path_or_buf=file_out)

# Write table with F1 scores
file_out = os.path.join(out_result_dir,
                        'table_multigaze_overfeat.csv')
data_F1 = np.asarray(all_f1_ksp_overfeat)
F1_means_all = np.mean(data_F1[:,[0,2,4,6]], axis=1).reshape(-1,1)
F1_std_all = np.mean(data_F1[:,[1,3,5,7]], axis=1).reshape(-1,1)
data_F1_overfeat = np.hstack((data_F1, F1_means_all, F1_std_all ))
print('Writing: ' + file_out)
df_ksp_overfeat = pd.DataFrame(data=data_F1_overfeat, index=I, columns=C)
df_ksp_overfeat.to_csv(path_or_buf=file_out)

#Write all in multi-index
methods = ['L2prior', 'L2', 'overfeat']
tuples = [(t,m) for t in rd.types for m in methods]
index = pd.MultiIndex.from_tuples(tuples, names=['Type', 'Loss'])
C = pd.Index(['0_mean','0_std',
              '1_mean','1_std',
              '2_mean','2_std',
              '3_mean','3_std',
              'all_mean', 'all_std'], name="columns")
data_F1_all = np.asarray([(data_F1_op[i,:], data_F1_L2[i,:], data_F1_overfeat[i,:]) for i in range(data_F1_L2.shape[0])]).reshape((12,10))

data_F1_all = np.round_(data_F1_all, decimals = n_decs)
is_max = []
for i in np.arange(0, int(data_F1_all.shape[0]), 2):
    if(np.argmax(data_F1_all[i:i+2,-2]) == 0):
        is_max.append((1,0))
    else:
        is_max.append((0,1))

df_all = pd.DataFrame(data=data_F1_all, index=index, columns=C)

cols_bold = ['0_dobold', '1_dobold', '2_dobold', '3_dobold', 'all_dobold']
for c in cols_bold:
    df_all[c] = False

# Bold on average on all sequences
for t in rd.types:
    for s in ['0', '1','2','3', 'all']:
        max_mean_ind = np.argmax(df_all.loc[t][s+'_mean'])
        df_all.loc[pd.IndexSlice[t, max_mean_ind], pd.IndexSlice[s+'_dobold']] = True

df_all = df_all.reindex_axis(sorted(df_all.columns), axis=1)


file_out = os.path.join(out_result_dir,
                        'ksp_losses.csv')
print('Writing: ' + file_out)
df_all.to_csv(path_or_buf=file_out)
