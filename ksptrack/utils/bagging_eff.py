import scipy
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import my_utils as utls
import progressbar
from skimage import (color, io, segmentation)
from sklearn import (mixture, metrics, preprocessing, decomposition)
from scipy import ndimage
import gc
import multiprocessing
import ctypes


def calc_bagging(T,
                 bag_max_depth,
                 bag_n_feats,
                 marked_arr,
                 marked_feats=None,
                 all_feats_df=None,
                 mode='foreground',
                 feat_fields=['desc'],
                 remove_marked=False,
                 bag_max_samples=2000):
    #marked_arr: has index of frame and corresponding superpixel label. Taken as positive samples
    #all_feats_df: Pandas frame with all samples (positive and unlabeled)
    #feat_fields: List of feature names as appearing in all_feats_df. Will be concatenated
    #labels: Superpixel labels
    #remove_marked (boolean): If True, marked SPs will be removed from all_feats_df

    #Load feats into shared memory object
    shared_array_base = multiprocessing.Array(ctypes.c_float,
                                              all_feats_df.shape[0]*all_feats_df[feat_fields].shape[1])
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(10, 10)

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
    #print('max features : ' + str(np.round(data_P.shape[1]*bag_n_feats)))
    #print('max depth: ' + str(bag_max_depth))

    NP = np.min((data_P.shape[0], bag_max_samples))
    NU = data_U.shape[0]

    np.random.seed(0)

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
            if(data_P.shape[0] > bag_max_samples):
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
                bag_max_depth=bag_max_depth,
                criterion='gini',
                splitter='best',
                presort=True,
                max_features = bag_n_feats,
                class_weight='balanced')
            model.fit(data_bootstrap, train_label)
            # Index for the out of the bag (oob) samples
            idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
            # Transductive learning of oob samples
            f_oob[idx_oob] += model.predict_proba(data_U[idx_oob,2:])
            n_oob[idx_oob] += 1
            predict_proba = f_oob[:, 1] / n_oob

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
