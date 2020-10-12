import time
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def fit_trees(args):
    #T, train_label, data_U, data_P, bag_max_depth, bag_n_feats, bag_max_samples = args
    T, train_label, bag_max_depth, bag_n_feats, bag_max_samples = args

    #print('started fit_trees on thread {}'.format(thread_id))
    NU = data_U.shape[0]
    #NP = data_P_shared.shape[0]
    NP = np.min((data_P.shape[0], bag_max_samples))
    K = NP

    #bs_p_idx = np.arange(data_P_shared.shape[0])
    bs_p_idx = np.random.choice(np.arange(data_P.shape[0]),
                                replace=True,
                                size=NP)

    n_oob = np.zeros(shape=(NU, ))
    f_oob = np.zeros(shape=(NU, 2))

    for t in range(T):
        bs_u_idx = np.random.choice(np.arange(data_U.shape[0]),
                                    replace=True,
                                    size=K)
        data_bootstrap = np.concatenate(
            (data_P[bs_p_idx, :], data_U[bs_u_idx, :]), axis=0)

        model = DecisionTreeClassifier(max_depth=bag_max_depth,
                                       max_features=bag_n_feats)

        model.fit(data_bootstrap, train_label)
        idx_oob = sorted(
            set(range(data_U.shape[0])) - set(np.unique(bs_u_idx)))
        f_oob[idx_oob] += model.predict_proba(data_U[idx_oob, :])
        n_oob[idx_oob] += 1
        predict_proba = f_oob[:, 1] / np.clip(n_oob, a_min=1, a_max=T)

    return predict_proba


def calc_logistic(feats, class_labels):

    estimator = LogisticRegression()
    # pu_estimator = PUAdapter(estimator, hold_out_ratio=0.)
    print('fitting PU logistic regression on {} samples'.format(
        feats.shape[0]))
    print('with {} positive samples'.format(class_labels.sum()))
    estimator.fit(feats, class_labels)
    probas = estimator.predict_proba(feats)
    return probas[:, 1], estimator.coef_


def calc_bagging(feats,
                 class_labels,
                 T,
                 bag_max_depth,
                 bag_n_feats,
                 bag_max_samples=500,
                 n_jobs=1):
    #marked_arr: has index of frame and corresponding superpixel label. Taken as positive samples
    #all_feats_df: Pandas frame with all samples (positive and unlabeled)
    #feat_fields: List of feature names as appearing in all_feats_df. Will be concatenated
    #labels: Superpixel labels
    #remove_marked (boolean): If True, marked SPs will be removed from all_feats_df

    #Remove positive samples from all_feats_df
    global data_U
    global data_P
    data_U = feats[np.logical_not(class_labels)]
    data_P = feats[class_labels]

    NP = np.min((data_P.shape[0], bag_max_samples))
    #NP = data_P.shape[0]
    NU = data_U.shape[0]

    print('number of positives samples: {}'.format(NP))
    print('max tree depth: {}'.format(bag_max_depth))
    print('max num feats: {}'.format(bag_n_feats))
    print('input features dims: {}'.format(data_U.shape[1]))

    np.random.seed(0)

    K = NP
    train_label = np.zeros(shape=(NP + K, ))
    train_label[:NP] = 1.0

    T_per_jobs = int(T / n_jobs)
    print('Will spawn {} job(s) with {} trees each'.format(n_jobs, T_per_jobs))

    t_start = time.time()
    if (n_jobs > 1):
        args = (T_per_jobs, train_label, bag_max_depth, bag_n_feats,
                bag_max_samples)
        args = [args for i in range(n_jobs)]
        pool = Pool()
        predict_probas = pool.map(fit_trees, args)

        predict_proba = np.mean(np.asarray(predict_probas), axis=0)
    else:
        predict_proba = fit_trees((T_per_jobs, train_label, bag_max_depth,
                                   bag_n_feats, bag_max_samples))

    elapsed = time.time() - t_start
    print('Done estimation in {:.2f} seconds.'.format(elapsed))

    probas = np.zeros(feats.shape[0])
    probas[np.logical_not(class_labels)] = predict_proba
    probas[class_labels] = 1.

    return probas


def merge_feats(df, feat_fields):
    out = []
    for i in df.index:
        out.append(
            np.hstack((df.frame[i], df.label[i],
                       np.hstack([
                           df[feat_field][i].reshape(1, -1)
                           for feat_field in feat_fields
                       ]).ravel())))
    out = np.asarray(out)

    return out
