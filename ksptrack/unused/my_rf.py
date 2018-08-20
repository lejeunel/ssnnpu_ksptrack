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



def run(X_train, y_train, X_test, T):

    """
    T: Number of decision trees to train
    X_train: Feature vectors
    y_train: Targets
    X_test: Feature vectors for evaluation
    """

    n_pos = np.sum(y_train)
    n_neg = np.sum(y_train == 0)

    ratio = 1.
    K_neg = int(n_pos*ratio)
    K_pos = int(n_pos*ratio)
    train_label = np.zeros(shape=(K_pos + K_neg, ))
    train_label[:K_pos] = 1.0
    n_oob = np.zeros(shape=(X_test.shape[0], ))
    f_oob = np.zeros(shape=(X_test.shape[0], 2))

    X_train_pos = X_train[y_train == 1,:]
    X_train_neg = X_train[y_train == 0,:]
    with progressbar.ProgressBar(maxval=T) as bar:
        for i in range(T):
            bar.update(i)
            # Bootstrap resample
            #bootstrap_sample_p = np.arange(n_pos)
            bootstrap_sample_p = np.random.choice(
                np.arange(n_pos), replace=True, size=K_pos)
            bootstrap_sample_n = np.random.choice(
                np.arange(n_neg), replace=True, size=K_neg)
            # Positive set + bootstrapped unlabeled set
            data_bootstrap = np.concatenate(
                (X_train_pos[bootstrap_sample_p,:], X_train_neg[bootstrap_sample_n, :]), axis=0)
            # Train model
            model = DecisionTreeClassifier(
                max_depth=15,
                max_features='sqrt',
                #max_features=50,
                #max_features=None,
                criterion='gini',
                splitter='best',
                #min_samples_split=100,
                #presort=True,
                #class_weight='balanced'
            )
            model.fit(data_bootstrap, train_label)

            # Transductive learning of oob samples
            f_oob += model.predict_proba(X_test)
            n_oob += 1
            predict_proba = f_oob[:, 1] / n_oob

    return predict_proba
