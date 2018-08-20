import scipy
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from . import my_utils as utls
import progressbar
from skimage import (color, io, segmentation)
from sklearn import (mixture, metrics, preprocessing, decomposition)
from scipy import ndimage
import gc
from multiprocessing import Pool
import sharedmem
from functools import partial
import time


def merge_feats(df,feat_fields):
    out = []
    for i in df.index:
            out.append(np.hstack((df.frame[i],
                                  df.sp_label[i],
                                  np.hstack([df[feat_field][i].reshape(1,-1) for feat_field in feat_fields]).ravel())))
    out = np.asarray(out)

    return out

def make_samples(marked_arr,
                 marked_feats=None, all_feats_df=None,
                 mode='foreground',
                 feat_fields=['desc'],
                 remove_marked=False):


    if(marked_feats is None):
        print("Extracting marked features")

        marked_feats = []
        with progressbar.ProgressBar(maxval=int(marked_arr.shape[0])) as bar:
            for i in range(marked_arr.shape[0]):

                bar.update(i)
                if(not np.any(marked_arr[i,:] == -1)): #Some points might be missing
                    this_idx = np.where((all_feats_df['frame']==marked_arr[i,0]) & (all_feats_df['sp_label']==marked_arr[i,1]) )[0][0]

                    this_feat = np.hstack([all_feats_df[feat_field][this_idx].reshape(1,-1)
                                            for feat_field in feat_fields]).ravel()
                    this_feat = np.hstack((all_feats_df.frame[this_idx],all_feats_df.sp_label[this_idx],this_feat))
                    marked_feats.append(this_feat)
            marked_feats = np.asarray(marked_feats)


    #Remove positive samples from all_feats_df
    data_U_df = all_feats_df
    if(remove_marked):
        for i in range(marked_arr.shape[0]):
            data_U_df = data_U_df[~((data_U_df.frame == marked_arr[i,0]) & (data_U_df.sp_label == marked_arr[i,1])) ]


    data_U = merge_feats(data_U_df,feat_fields=feat_fields)
    #data_U = data_U[:,2:]

    #In case of background, some frames have no marked SP, thus need to filter them out
    nan_idx = np.sum(np.isnan(marked_feats),axis=1).astype(bool)
    data_P = marked_feats[~nan_idx,:]

    return data_U, data_P

def calc_bagging(T,
                 max_depth,
                 max_n_feats,
                 marked_arr,
                 marked_feats=None,
                 all_feats_df=None,
                 mode='foreground',
                 feat_fields=['desc'],
                 remove_marked=False,
                 max_samples=2000):
    #marked_arr: has index of frame and corresponding superpixel label. Taken as positive samples
    #all_feats_df: Pandas frame with all samples (positive and unlabeled)
    #feat_fields: List of feature names as appearing in all_feats_df. Will be concatenated
    #labels: Superpixel labels
    #remove_marked (boolean): If True, marked SPs will be removed from all_feats_df


    if(marked_feats is None):
        print("Extracting marked features")

        marked_feats = []
        with progressbar.ProgressBar(maxval=int(marked_arr.shape[0])) as bar:
            for i in range(marked_arr.shape[0]):

                bar.update(i)
                if(not np.any(marked_arr[i,:] == -1)): #Some points might be missing
                    this_idx = np.where((all_feats_df['frame']==marked_arr[i,0]) & (all_feats_df['sp_label']==marked_arr[i,1]) )[0][0]

                    this_feat = np.hstack([all_feats_df[feat_field][this_idx].reshape(1,-1)
                                            for feat_field in feat_fields]).ravel()
                    this_feat = np.hstack((all_feats_df.frame[this_idx],all_feats_df.sp_label[this_idx],this_feat))
                    marked_feats.append(this_feat)
            marked_feats = np.asarray(marked_feats)


    #Remove positive samples from all_feats_df
    data_U_df = all_feats_df
    if(remove_marked):
        for i in range(marked_arr.shape[0]):
            data_U_df = data_U_df[~((data_U_df.frame == marked_arr[i,0]) & (data_U_df.sp_label == marked_arr[i,1])) ]


    data_U = merge_feats(data_U_df,feat_fields=feat_fields)

    #In case of background, some frames have no marked SP, thus need to filter them out
    nan_idx = np.sum(np.isnan(marked_feats),axis=1).astype(bool)
    data_P = marked_feats[~nan_idx,:]

    print('number of positives samples: ' + str(data_P.shape[0]))
    #print('max features : ' + str(np.round(data_P.shape[1]*max_n_feats)))
    #print('max depth: ' + str(max_depth))

    NP = np.min((data_P.shape[0], max_samples))
    NU = data_U.shape[0]

    np.random.seed(0)

    t_start = time.time()

    print('learning')
    K = NP
    train_label = np.zeros(shape=(NP + K, ))
    train_label[:NP] = 1.0
    n_oob = np.zeros(shape=(NU, ))
    f_oob = np.zeros(shape=(NU, 2))
    with progressbar.ProgressBar(maxval=T) as bar:
        for i in range(T):
            bar.update(i)
            # Bootstrap resample
            if(data_P.shape[0] > max_samples):
                bootstrap_sample_p = np.random.choice(
                    np.arange(data_P.shape[0]), replace=True, size=NP)
            else:
                bootstrap_sample_p = np.arange(data_P.shape[0])
            bootstrap_sample = np.random.choice(
                np.arange(NU), replace=True, size=K)
            # Positive set + bootstrapped unlabeled set
            data_bootstrap = np.concatenate(
                (data_P[bootstrap_sample_p,2:], data_U[bootstrap_sample, 2:]), axis=0)
            # Train model
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion='gini',
                splitter='best',
                presort=True,
                max_features = max_n_feats,
                class_weight='balanced')
            model.fit(data_bootstrap, train_label)
            # Index for the out of the bag (oob) samples
            idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
            # Transductive learning of oob samples
            f_oob[idx_oob] += model.predict_proba(data_U[idx_oob,2:])
            n_oob[idx_oob] += 1
            predict_proba = f_oob[:, 1] / n_oob

    elapsed = time.time() - t_start
    print('Done estimation in {} seconds.'.format(elapsed))

    #Concatenate probas to data_U
    if(not remove_marked):
        data_frames = data_U[:,0].reshape(-1,1).astype(int)
        data_labels = data_U[:,1].reshape(-1,1).astype(int)
        data_probas = predict_proba.reshape(-1,1)

    else:
        data_frames = np.vstack((data_U[:,0].reshape(-1,1), data_P[:,0].reshape(-1,1))).astype(int)
        data_labels = np.vstack((data_U[:,1].reshape(-1,1), data_P[:,1].reshape(-1,1))).astype(int)
        data_probas = np.vstack((predict_proba.reshape(-1,1), np.ones((data_P.shape[0],1))))

    pm_df = pd.DataFrame({'frame':data_frames.ravel(),'sp_label':data_labels.ravel(),'proba':data_probas.ravel()})

    pm_df.sort_values(['frame','sp_label'],inplace=True)

    return marked_feats, pm_df


def calc_bagging_para(T,
                      max_depth,
                      max_n_feats,
                      marked_arr,
                      marked_feats=None,
                      all_feats_df=None,
                      mode='foreground',
                      feat_fields=['desc'],
                      remove_marked=False,
                      max_samples=2000,
                      n_jobs=4):
    #marked_arr: has index of frame and corresponding superpixel label. Taken as positive samples
    #all_feats_df: Pandas frame with all samples (positive and unlabeled)
    #feat_fields: List of feature names as appearing in all_feats_df. Will be concatenated
    #labels: Superpixel labels
    #remove_marked (boolean): If True, marked SPs will be removed from all_feats_df


    if(marked_feats is None):
        print("Extracting marked features")

        marked_feats = []
        #with progressbar.ProgressBar(maxval=int(marked_arr.shape[0])) as bar:
        for i in range(marked_arr.shape[0]):

            #bar.update(i)
            #print(i)
            #if(i == 28):
            #    import pdb; pdb.set_trace()

            if(not np.any(marked_arr[i,:] == -1)): #Some points might be missing
                this_idx = np.where((all_feats_df['frame']==marked_arr[i,0])\
                        & (all_feats_df['sp_label']==marked_arr[i,1]) )[0][0]

                this_feat = np.hstack([
                        all_feats_df[feat_field].iloc[this_idx].reshape(1,-1)
                                        for feat_field in feat_fields
                    ]).ravel()
                this_feat = np.hstack((all_feats_df.frame.iloc[this_idx],
                                       all_feats_df.sp_label.iloc[this_idx],
                                       this_feat))
                marked_feats.append(this_feat)
        marked_feats = np.asarray(marked_feats)


    #Remove positive samples from all_feats_df
    data_U_df = all_feats_df
    if(remove_marked):
        for i in range(marked_arr.shape[0]):
            data_U_df = data_U_df[~((data_U_df.frame == marked_arr[i,0]) & (data_U_df.sp_label == marked_arr[i,1])) ]


    data_U = merge_feats(data_U_df,feat_fields=feat_fields)

    #In case of background, some frames have no marked SP, thus need to filter them out
    nan_idx = np.sum(np.isnan(marked_feats),axis=1).astype(bool)
    data_P = marked_feats[~nan_idx,:]

    data_U_shared = sharedmem.copy(data_U)
    data_P_shared = sharedmem.copy(data_P)

    print('number of positives samples: ' + str(data_P.shape[0]))

    NP = np.min((data_P.shape[0], max_samples))
    #NP = data_P.shape[0]
    NU = data_U.shape[0]

    np.random.seed(0)

    K = NP
    train_label = np.zeros(shape=(NP + K, ))
    train_label[:NP] = 1.0

    T_per_process = int(T/n_jobs)
    print('Will spawn {} processes with {} trees per process'.format(n_jobs,T_per_process))

    t_start = time.time()
    with sharedmem.MapReduce(np=n_jobs) as pool:
        def fit_trees(args):

            T, train_label, max_depth, max_n_feats, max_samples = args

            NU = data_U_shared.shape[0]
            #NP = data_P_shared.shape[0]
            NP = np.min((data_P_shared.shape[0], max_samples))
            K = NP

            #bs_p_idx = np.arange(data_P_shared.shape[0])
            bs_p_idx = np.random.choice(
                np.arange(data_P_shared.shape[0]), replace=True, size=NP)

            n_oob = np.zeros(shape=(NU, ))
            f_oob = np.zeros(shape=(NU, 2))

            for t in range(T):
                bs_u_idx = np.random.choice(
                    np.arange(data_U_shared.shape[0]), replace=True, size=K)
                data_bootstrap = np.concatenate(
                    (data_P_shared[bs_p_idx, 2:], data_U_shared[bs_u_idx, 2:]), axis=0)

                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    criterion='gini',
                    splitter='best',
                    presort=True,
                    max_features = max_n_feats,
                    class_weight='balanced')

                model.fit(data_bootstrap, train_label)
                idx_oob = sorted(set(range(data_U_shared.shape[0])) - set(np.unique(bs_u_idx)))
                f_oob[idx_oob] += model.predict_proba(data_U_shared[idx_oob,2:])
                n_oob[idx_oob] += 1
                predict_proba = f_oob[:, 1] / n_oob

            return predict_proba

        def reduce(r):
            print('job done.')
            return r

        args = (T_per_process,
                train_label,
                max_depth,
                max_n_feats,
                max_samples)
        args = [args for i in range(n_jobs)]
        predict_probas = pool.map(fit_trees, args, reduce=reduce)

    #print(predict_probas)
    predict_proba = np.mean(np.asarray(predict_probas), axis=0)

    elapsed = time.time() - t_start
    print('Done estimation in {} seconds.'.format(elapsed))
    #Concatenate probas to data_U
    if(not remove_marked):
        data_frames = data_U[:,0].reshape(-1,1).astype(int)
        data_labels = data_U[:,1].reshape(-1,1).astype(int)
        data_probas = predict_proba.reshape(-1,1)

    else:
        data_frames = np.vstack((data_U[:,0].reshape(-1,1), data_P[:,0].reshape(-1,1))).astype(int)
        data_labels = np.vstack((data_U[:,1].reshape(-1,1), data_P[:,1].reshape(-1,1))).astype(int)
        data_probas = np.vstack((predict_proba.reshape(-1,1), np.ones((data_P.shape[0],1))))

    pm_df = pd.DataFrame({'frame':data_frames.ravel(),'sp_label':data_labels.ravel(),'proba':data_probas.ravel()})

    pm_df.sort_values(['frame','sp_label'],inplace=True)

    return marked_feats, pm_df
