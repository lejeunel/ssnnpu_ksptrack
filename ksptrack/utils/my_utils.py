import datetime
import glob
import logging
import logging.config
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import tqdm
import yaml
from scipy import interpolate
from skimage import color, io
from sklearn.metrics import f1_score

import ksptrack
from ksptrack.utils import bagging as bag

from . import csv_utils as csv


def probas_to_df(labels, probas):

    if (isinstance(probas, list)):
        probas = np.concatenate(probas)
    if (probas.ndim == 1):
        probas = probas[..., None]

    frame_labels_list = [[(f, l) for f in range(labels.shape[0])
                          for l in np.unique(labels[f])]]
    frame_labels = np.squeeze(np.array(frame_labels_list))
    frame_labels_probas = np.hstack((frame_labels, probas))
    probas = pd.DataFrame(frame_labels_probas,
                          columns=['frame', 'label', 'proba'])
    probas = probas.astype({'frame': int, 'label': int})

    return probas


def get_pm_array(labels, probas):
    """ Returns array same size as labels with probabilities of bagging model
    """

    if (not isinstance(probas, pd.DataFrame)):
        probas = probas_to_df(labels, probas)

    scores = labels.copy().astype(float)
    frames = np.arange(scores.shape[0])

    i = 0
    bar = tqdm.tqdm(total=len(frames))
    for f in frames:
        this_frame_pm_df = probas[probas['frame'] == f]
        dict_keys = this_frame_pm_df['label']
        dict_vals = this_frame_pm_df['proba']
        dict_map = dict(zip(dict_keys, dict_vals))
        # Create 2D replacement matrix
        replace = np.array([list(dict_map.keys()), list(dict_map.values())])
        # Find elements that need replacement
        mask = np.isin(scores[f], replace[0, :])
        # Replace elements
        scores[f, mask] = replace[1,
                                  np.searchsorted(replace[0, :], scores[f,
                                                                        mask])]
        i += 1
        bar.update(1)
    bar.close()

    return scores


def get_binary_array(labels, sps):
    """ Returns boolean array same size as labels with positive superpixels set to 1
    """

    scores = np.zeros_like(labels).astype(bool)
    frames = np.arange(scores.shape[0])

    bar = tqdm.tqdm(total=len(frames))
    for f in frames:
        l = np.unique(sps[sps[:, 0] == f][:, 1])
        scores[f] = np.sum([labels[f] == l_ for l_ in l], axis=0)
        bar.update(1)
    bar.close()

    return scores


def sample_features(X, y, threshs, n_samp, check_thr=True, n_bins=None):
    """
    X: features matrix
    y: probability values
    thresh: [low_thr, high_thr]
    n_samp: number of samples to select randomly from each class
    """

    # Check for dimensions with all zeros
    # inds_ = np.where(np.sum(X, axis=0) != 0)[0]
    # X = X[:, inds_]

    if (check_thr):
        threshs = check_thrs(threshs, y, n_samp)

    idx_neg = np.where(y < threshs[0])[0]
    idx_pos = np.where(y >= threshs[1])[0]

    p_pos = None
    p_neg = None

    if (n_bins is not None):
        bins_neg = np.linspace(0, threshs[0], n_bins)
        bins_pos = np.linspace(threshs[1], 1, n_bins)
        idx_bins_neg = np.digitize(y[idx_neg], bins_neg, right=True)
        idx_bins_pos = np.digitize(y[idx_pos], bins_pos, right=True)
        freqs_neg = np.bincount(idx_bins_neg)
        freqs_pos = np.bincount(idx_bins_pos)
        weights_pos = idx_pos.size / freqs_pos[idx_bins_pos]
        weights_neg = idx_neg.size / freqs_neg[idx_bins_neg]
        p_pos = weights_pos / weights_pos.sum()
        p_neg = weights_neg / weights_neg.sum()

    idx_pos = np.random.choice(idx_pos, size=n_samp, p=p_pos)
    idx_neg = np.random.choice(idx_neg, size=n_samp, p=p_neg)

    X_pos = X[idx_pos, :]
    X_neg = X[idx_neg, :]
    y_pos = y[idx_pos]
    y_neg = y[idx_neg]
    descs = np.concatenate((X_pos, X_neg), axis=0)

    y = np.concatenate((np.ones_like(y_pos), np.zeros_like(y_neg)), axis=0)

    return descs, y


def df_crossjoin(df1, df2, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


def my_interp(x, y, n_samp, xrange_=[0, 1]):

    x = x.ravel()
    y = y.ravel()
    x_out = np.linspace(x[0], x[-1], n_samp)
    #x_out = np.linspace(xrange_[0],xrange_[1],n_samp)
    f = interpolate.interp1d(x.ravel(), y.ravel())

    try:
        y_out = f(x_out)
    except:
        x_out = np.linspace(xrange_[0], xrange_[1], n_samp)
        #x_out = np.linspace(x[0],x[-1],n_samp)
        f = interpolate.interp1d(x.ravel(), y.ravel())
        y_out = f(x_out)

    return x_out, y_out


def concat_interp(x, y, n_samp):

    y_out = []

    for i in range(len(y)):
        this_x_out, this_y_out = my_interp(x[i], y[i], n_samp)
        y_out.append(this_y_out)

    x_out = this_x_out

    return x_out, np.asarray(y_out)


def concat_arr(x):

    x_out = []

    for i in range(x.shape[0]):
        x_out.append(x[i])

    return np.asarray(x_out)


def make_y_array_true(map_, labels, pos_thr=0.5):

    #self.logger.info('Making Y vector (true groundtruths)')
    y = []
    bar = tqdm.tqdm(total=labels.shape[-1])
    for i in range(labels.shape[-1]):
        bar.update(1)
        for j in np.unique(labels[..., i]):
            this_mask = labels[..., i] == j
            this_overlap = np.logical_and(map_[..., i], this_mask)
            y.append(
                (i, j, np.sum(this_overlap) / np.sum(this_mask) > pos_thr))

    return np.asarray(y)


def make_sp_gts(map_, labels, pos_thr=0.5):

    #Convert superpixel labels to ground-truth based on threshold
    y = np.zeros(labels.shape, dtype=np.bool)

    with progressbar.ProgressBar(maxval=labels.shape[-1]) as bar:
        for i in range(labels.shape[-1]):
            #for i in range(1):
            bar.update(i)
            for j in np.unique(labels[..., i]):
                mask = labels[..., i] == j
                overlap = np.logical_and(map_[..., i], mask)
                if (np.sum(overlap) / np.sum(mask) > pos_thr):
                    y[..., i] += mask

    return y


def setup_logging(log_path,
                  conf_path='logging.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = conf_path

    # Get absolute path to logging.yaml
    path = os.path.join(os.path.dirname(ksptrack.__file__), path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            config['handlers']['info_file_handler']['filename'] = os.path.join(
                log_path, 'info.log')
            config['handlers']['error_file_handler'][
                'filename'] = os.path.join(log_path, 'error.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def split_homogeneous(arr, vals):
    """
    arr : array with values to split
    vals: array of values that determines homogeneity
    ---
    Returns:
    inds: list of indices with each element having indices of homogeneous chunk
    """

    out = []
    diff = vals[1:] - vals[0:-1]
    split_ind = np.where(diff > 0)

    init_ind = np.arange(vals.shape[0])

    start = 0
    while (split_ind.size > 0):
        chunk = init_ind[start:split_ind[0]]
        out.append(chunk)
        start = split_ind[0] + 1
        split_ind = split_ind[1:]

    return out


def make_sc_graph(R, F):

    g = nx.DiGraph()
    keys = R.keys()

    labels_per_frame = np.asarray([np.unique(F[i]) for i in range(len(F))])
    all_labels = np.unique(np.concatenate(labels_per_frame))

    #Add nodes first with corresponding scale (lowest = 0)
    #for i in range(all_labels.shape[0]):
    #    g.add_node(all_labels[i],stack_idx=np.min([s for s in range(len(labels_per_frame)) if(all_labels[i] in labels_per_frame[s])]))

    non_empty_keys = [k for k in keys if (len(R[k]) > 0)]

    root = non_empty_keys[-1]

    stack_idx = dict(nx.get_node_attributes(g, 'stack_idx'))

    #Link and store scale difference (parent - child)
    for k in non_empty_keys:
        frame_parent_idx = np.min([
            s for s in range(len(labels_per_frame))
            if (k in labels_per_frame[s])
        ])
        frame_child_left = np.max([
            s for s in range(len(labels_per_frame))
            if (R[k][0] in labels_per_frame[s])
        ])
        frame_child_right = np.max([
            s for s in range(len(labels_per_frame))
            if (R[k][1] in labels_per_frame[s])
        ])
        #ds = stack_idx[k] - stack_idx[R[k][j]]
        g.add_edge(k, R[k][0], ds=frame_parent_idx - frame_child_left)
        g.add_edge(k, R[k][1], ds=frame_parent_idx - frame_child_right)

    scales = nx.single_source_dijkstra_path_length(g, root, weight='ds')

    return g, scales


def merge_feats(df, feat_fields):
    out = []
    for i in df.index:
        out.append(
            np.hstack((df.frame[i], df.sp_label[i],
                       np.hstack([
                           df[feat_field][i].reshape(1, -1)
                           for feat_field in feat_fields
                       ]).ravel())))

    del df
    return np.asarray(out)


def ksp2array(g,
              ksp_dict,
              labels,
              arg_direction=['forward_sets', 'backward_sets']):
    #Flatten all edges and make node list. Returns array (frame_ind, sp_label)

    nodes = []
    for direction in arg_direction:
        if direction in ksp_dict.keys():
            this_set = ksp_dict[direction][-1]
            this_paths = [item for sublist in this_set for item in sublist]
            this_nodes = [
                n for n in this_paths if ((n != g.source) and (n != g.sink))
            ]
            this_nodes = [
                n[0] for n in this_nodes
                if ((n[1] == g.nodeTypes.index('input')))
            ]
        nodes += this_nodes

    #Keep only uniques
    nodes = np.asarray(list(set(nodes)))

    idx_sort = np.argsort(nodes[:, 0])
    return nodes[idx_sort, :]


def hist_inter(x, y, w=None):

    if (w is not None):
        out = np.min(w * np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1),
                     axis=1)
    else:
        out = np.min(np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1),
                     axis=1)
    return np.sum(out)


def hoof(f, nbins):

    bins = np.linspace(-np.pi, np.pi, nbins)
    angle = np.arctan2(f[:, 1], f[:, 0])
    norm = np.linalg.norm(f, axis=1)
    idx = np.digitize(angle, bins)

    out = np.zeros(bins.shape[0])
    for i in range(norm.shape[0]):
        out[idx[i]] += norm[i]

    sum_out = np.sum(out)
    if (sum_out == 0):
        return out
    else:
        return out / np.sum(out)


def get_complement_arrays(arr):

    out = np.empty(arr.shape)
    for i in range(arr.shape[2]):
        out[..., i] = np.logical_not(arr[..., i])

    return out


def get_complement(dict_ksp, labels, source, sink, node_in_num=0):
    #Return complement of node set (for later computation of background model)

    #Get all nodes from labels
    all_nodes = []
    for i in range(labels.shape[0]):
        for j in np.unique(labels[i, :, :]):
            all_nodes.append((i, j))

    #Get nodes from ksp set
    nodes = []
    for direction in {'forward_sets', 'backward_sets'}:
        if direction in kspSet.keys():
            this_set = dict_ksp[direction][-1]
            this_paths = [item for sublist in this_set for item in sublist]
            this_nodes = [
                n for n in this_paths if ((n != source) and (n != sink))
            ]
            this_nodes = [n[0] for n in this_nodes if ((n[1] == node_in_num))]
            nodes += this_nodes

    #Remove duplicates (two directions)
    list(set(nodes))

    sorted_nodes = sorted(nodes, key=lambda tup: tup[0])

    complement_nodes = [
        all_nodes[i] for i in range(len(all_nodes))
        if (all_nodes[i] not in nodes)
    ]

    return complement_nodes


def get_scores_ksp_tracklets(dict_ksp,
                             winInd,
                             frameInd,
                             frameFileNames,
                             labels,
                             Kmax=None,
                             mode='all'):

    scores = score_path_tracklets(dict_ksp,
                                  labels,
                                  's',
                                  't',
                                  set_idx=winInd,
                                  frame_idx=frameInd,
                                  Kmax=Kmax,
                                  mode=mode).astype(int)

    return scores


def get_scores_from_sps(sps_arr, labels, probas=None):

    scores = np.zeros(labels.shape)
    if (probas is None):
        probas = np.ones(sps_arr.shape[0]).astype(int)

    n = 0
    with progressbar.ProgressBar(maxval=sps_arr.shape[0]) as bar:
        for i in range(sps_arr.shape[0]):
            bar.update(i)
            scores[..., sps_arr[i, 0]] += (labels[:, :, sps_arr[i, 0]]
                                           == sps_arr[i, 1]) * probas[i]

    return scores


def get_scores_ksp(dict_ksp,
                   winInd,
                   frameInd,
                   gt_dir,
                   frameFileNames,
                   labels,
                   Kmax=None,
                   mode='all'):

    #Extract ground-truth files
    gt = np.zeros((len(frameFileNames), labels.shape[0], labels.shape[1]))
    for i in range(len(frameFileNames)):
        base, fname = os.path.split(frameFileNames[i])
        this_gt = imread(os.path.join(gt_dir, fname))
        gt[i, :, :] = (this_gt[:, :, 0] > 0)

    #scores = np.zeros((len(frameFileNames),labels.shape[0],labels.shape[1]))
    scores = score_path_sets(dict_ksp,
                             labels,
                             's',
                             't',
                             set_idx=winInd,
                             frame_idx=frameInd,
                             Kmax=Kmax,
                             mode=mode).astype(int)

    return scores


def readCsv(csvName, seqStart=None, seqEnd=None):

    out = np.loadtxt(open(csvName, "rb"), delimiter=";",
                     skiprows=5)[seqStart:seqEnd, :]
    if ((seqStart is not None) or (seqEnd is not None)):
        out[:, 0] = np.arange(0, seqEnd - seqStart)

    return pd.DataFrame(data=out,
                        columns=['frame', 'time', 'visible', 'x', 'y'])


def getDataOutDir(dataOutRoot, ds_dir, resultDir, out_dir_prefix, testing):

    now = datetime.datetime.now()
    dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")

    dataOutDir = os.path.join(dataOutRoot, ds_dir, resultDir)
    dataOutResultDir = os.path.join(dataOutDir,
                                    dateTime + '_' + out_dir_prefix)

    #print(dataOutResultDir)
    if (not os.path.exists(dataOutResultDir)) and (not testing):
        os.makedirs(dataOutResultDir)

    return dataOutResultDir


def array_to_marked(arr, labels):
    marked = []
    for i in range(arr.shape[2]):
        this_mask = np.where(arr[..., i])
        this_marked = np.unique(labels[this_mask[0], this_mask[1], i])
        this_frame_col = np.tile(i, this_marked.shape[0])
        marked.append(
            np.concatenate(
                (this_frame_col.reshape(-1, 1), this_marked.reshape(-1, 1)),
                axis=1))

    out = np.concatenate(marked)

    return out


def marked_to_scores(marked, labels):

    scores = np.zeros(labels.shape)
    #for f in labels.shape[0]:
    frames = np.unique(marked[:, 0])
    for f in frames:
        sps = marked[marked[:, 0] == f, :]
        for s in range(sps.shape[0]):
            scores[..., f] += labels[..., f] == sps[s, 1]

    return scores


def sps2marked(P):

    marked = [m for sublist in P for m in sublist]
    return np.asarray(marked)


def aggressive_flatten(xs):
    result = []
    if isinstance(xs, list):
        for x in xs:
            result.extend(aggressive_flatten(x))
    else:
        result.append(xs)
    return result


def ksp2sps(paths, tls):
    #Flatten all edges and return list of sps

    ids = []
    for p in paths:
        #Keep only uniques
        ids.append([])
        ids[-1] = [i for i in p if (i != -1)]

    this_tls = []
    for p in ids:
        this_tls.append([])
        this_tls[-1] = [
            t for t in tls for i in range(len(p)) if (t.id_ == p[i])
        ]
        # sort tracklets according to direction
        dir_ = this_tls[-1][0].direction
        frames = [t.get_in_frame() for t in this_tls[-1]]
        arg_sort_frames = np.argsort(frames)
        if (dir_ == 'forward'):
            this_tls[-1] = [this_tls[-1][i] for i in arg_sort_frames]
        elif (dir_ == 'backward'):
            arg_sort_frames = arg_sort_frames[::-1]
            this_tls[-1] = [this_tls[-1][i] for i in arg_sort_frames]

    sps = []
    for p in this_tls:
        sps.append([])
        sps[-1] = aggressive_flatten([t.sps for t in p])

    return sps, this_tls


def ksp2array_tracklets(dict_ksp,
                        labels,
                        arg_direction=['forward_sets', 'backward_sets'],
                        source='s',
                        sink='t'):
    #Flatten all edges and make node list. Returns array (frame_ind, sp_label)

    sps = []
    for direction in arg_direction:
        if direction in dict_ksp.keys():
            if (direction == 'forward_sets'):
                this_tracklets = dict_ksp['forward_tracklets']
            else:
                this_tracklets = dict_ksp['backward_tracklets']

            this_set = dict_ksp[direction][-1]
            this_nodes = [item for sublist in this_set for item in sublist]
            this_nodes = [
                n for n in this_nodes if ((n != source) and (n != sink))
            ]
            this_ids = [n[0] for n in this_nodes]
            this_ids = list(set(this_ids))
            this_tls = [t for t in this_tracklets if (t.id_ in this_ids)]
            this_sps = [t.sps for t in this_tls]
            this_sps = [item for sublist in this_sps for item in sublist]
            #this_sps = [item for sublist in this_sps for item in sublist]
            sps += this_sps

    #Keep only uniques
    sps = [item for sublist in sps for item in sublist]
    sps = np.asarray(list(set(sps)))

    idx_sort = np.argsort(sps[:, 0])
    return sps[idx_sort, :]


def list_paths_to_seeds(forw, back, iter_=-1):

    back_sps = [p.tolist() for p in back[iter_]]
    back_sps = [tuple(item[0]) for sublist in back_sps for item in sublist]
    for_sps = [p.tolist() for p in forw[iter_]]
    for_sps = [tuple(item[0]) for sublist in for_sps for item in sublist]
    seeds = np.asarray(list(set(back_sps + for_sps)))

    return seeds


def get_node_list_tracklets(dict_ksp, source='s', sink='t', node_in_num=0):
    #Return list of nodes of last family of ksp sets
    sps = []
    for direction in {'forward_sets', 'backward_sets'}:
        if direction in dict_ksp.keys():
            if (direction == 'forward_sets'):
                this_tracklets = dict_ksp['forward_tracklets']
            else:
                this_tracklets = dict_ksp['backward_tracklets']

            this_set = dict_ksp[direction][-1]
            this_nodes = [item for sublist in this_set for item in sublist]
            this_nodes = [
                n for n in this_nodes if ((n != source) and (n != sink))
            ]
            this_ids = [n[0] for n in this_nodes]
            this_ids = list(set(this_ids))
            this_tls = [t for t in this_tracklets if (t.id_ in this_ids)]
            this_sps = [t.sps for t in this_tls]
            this_sps = [item for sublist in this_sps for item in sublist]
            this_sps = [item for sublist in this_sps for item in sublist]
            sps += this_sps

    #Remove duplicates (two directions)
    sps = list(set(sps))

    sorted_sps = sorted(sps, key=lambda tup: tup[0])

    return sorted_sps


def get_node_list(dict_ksp, source, sink, node_in_num=0):
    #Return list of nodes of last family of ksp sets

    nodes = []
    for direction in {'forward_sets', 'backward_sets'}:
        if direction in dict_ksp.keys():
            this_set = dict_ksp[direction][-1]
            this_paths = [item for sublist in this_set for item in sublist]
            this_nodes = [
                n for n in this_paths if ((n != source) and (n != sink))
            ]
            this_nodes = [n[0] for n in this_nodes if ((n[1] == node_in_num))]
            nodes += this_nodes

    #Remove duplicates (two directions)
    list(set(nodes))

    sorted_nodes = sorted(nodes, key=lambda tup: tup[0])

    return nodes


def sp_tuples_to_mat(P, labels):

    scores = np.zeros(labels.shape, dtype=bool)
    for p in P:
        for s in p:
            mask = labels[s[0]] == s[1]
            scores[s[0]] += mask

    return scores


def score_path_tracklets(kspSet,
                         labels,
                         source,
                         sink,
                         set_idx=None,
                         frame_idx=None,
                         node_in_num=0,
                         Kmax=None,
                         mode='stop'):

    if frame_idx is None:
        frame_idx = np.array([0])
    if set_idx is None:
        set_idx = 0

    if (type(frame_idx).__name__ != 'ndarray'):
        frame_array = np.array([frame_idx])
    else:
        frame_array = frame_idx

    if (Kmax is None):
        Kmax = np.NAN

    scores = np.zeros((labels.shape[0], labels.shape[1], frame_array.shape[0]))

    nodes = get_node_list_tracklets(kspSet, source, sink, node_in_num)

    with progressbar.ProgressBar(maxval=len(nodes)) as bar:
        for n in range(len(nodes)):

            bar.update(n)
            scores[..., nodes[n][0]] += labels[..., nodes[n][0]] == nodes[n][1]

    return scores


def make_full_y_from_ksp(dict_ksp, sps):

    sps_ksp = np.asarray(
        get_node_list_tracklets(dict_ksp, source='s', sink='t', node_in_num=0))

    seeds_out = np.concatenate((sps['frame'].reshape(
        -1, 1), sps['sp_label'].reshape(-1, 1), np.zeros((sps.shape[0], 1))),
                               axis=1)

    for i in range(sps_ksp.shape[0]):
        idx = np.where((seeds_out[:, 0] == sps_ksp[i, 0])
                       & (seeds_out[:, 1] == sps_ksp[i, 1]))[0]
        seeds_out[idx, -1] = 1

    return seeds_out


def seeds_to_scores(labels, seeds):
    scores = np.zeros(labels.shape)

    with progressbar.ProgressBar(maxval=seeds.shape[0]) as bar:
        for n in range(seeds.shape[0]):
            bar.update(n)
            scores[..., seeds[n, 0]] += labels[..., seeds[n, 0]] == seeds[n, 1]

    return scores


def score_path_sets(kspSet,
                    labels,
                    source,
                    sink,
                    set_idx=None,
                    frame_idx=None,
                    node_in_num=0,
                    Kmax=None,
                    mode='stop'):

    if frame_idx is None:
        frame_idx = np.array([0])
    if set_idx is None:
        set_idx = 0

    if (type(frame_idx).__name__ != 'ndarray'):
        frame_array = np.array([frame_idx])
    else:
        frame_array = frame_idx

    if (Kmax is None):
        Kmax = np.NAN

    scores = np.zeros((labels.shape[0], labels.shape[1], frame_array.shape[0]))

    nodes = get_node_list(kspSet, source, sink, node_in_num)

    with progressbar.ProgressBar(maxval=len(nodes)) as bar:
        for n in range(len(nodes)):
            bar.update(n)
            scores[..., nodes[n][0]] += labels[:, :,
                                               nodes[n][0]] == nodes[n][1]

    return scores


def imread(fname):

    img = io.imread(fname)

    if (img.dtype == np.uint16):
        img = ((img / (2**16)) * (2**8)).astype(np.uint8)

    if (len(img.shape) > 2):
        nchans = img.shape[2]
        if (nchans > 3):
            return img[:, :, 0:3]
        else:
            return img
    else:
        return color.gray2rgb(img)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def getPositiveSPMasks(g_arg, spLabels):

    posMasks = np.zeros(
        (len(spLabels), spLabels[0].shape[0], spLabels[0].shape[1]), np.uint8)

    for v in g_arg.vertices():
        mask = (spLabels[g_arg.vp["frameInd"][v]] == g_arg.vp["partInd"][v])
        posMasks[g_arg.vp["frameInd"][v], :, :] += mask

    return posMasks


def getPositives(groundTruthFileNames):

    positives = []

    for i in range(len(groundTruthFileNames)):
        img = imread(groundTruthFileNames[i])
        gt = img > 0
        positives.append(gt)

    positives = np.asarray(positives)
    positives = positives[:, :, :, 0]
    positives = np.transpose(positives, (1, 2, 0))
    return positives


def getTrueFalsePositiveRates(gt_positives, seg_positives, thr=None):

    if (thr is None):
        thr = np.asarray([1])

    truePositives = np.bool((gt_positives.shape[0]))
    falsePositives = np.bool((gt_positives.shape[0]))
    trueNegatives = np.bool((gt_positives.shape[0]))
    falseNegatives = np.bool((gt_positives.shape[0]))
    true_pos_rate = np.zeros((thr.shape[0]))
    false_pos_rate = np.zeros((thr.shape[0]))

    with progressbar.ProgressBar(maxval=thr.shape[0]) as bar:
        for i in range(thr.shape[0]):  #thresholds
            bar.update(i)
            #print "Calculating rates with score above: ", scores[j]
            truePositives = np.logical_and(gt_positives,
                                           seg_positives >= thr[i])
            falsePositives = np.logical_and(np.logical_not(gt_positives),
                                            seg_positives >= thr[i])
            trueNegatives = np.logical_and(
                np.logical_not(gt_positives),
                np.logical_not(seg_positives >= thr[i]))
            falseNegatives = np.logical_and(
                gt_positives, np.logical_not(seg_positives >= thr[i]))

            true_pos_rate[i] = float(np.sum(truePositives)) / float(
                np.sum(truePositives) + np.sum(falseNegatives))
            false_pos_rate[i] = float(np.sum(falsePositives)) / float(
                np.sum(falsePositives) + np.sum(trueNegatives))

    return (true_pos_rate, false_pos_rate)


def get_max_f_score(gt_positives, seg_positives, thr):
    f1 = []
    for t in thr:
        f1.append(
            f1_score(
                gt_positives.astype(int).ravel(),
                (seg_positives > t).astype(int).ravel()))

    return np.max(f1)


def conf_mat_to_tpr_fpr(conf_mat):

    tp = conf_mat[1, 1]
    fp = conf_mat[0, 1]
    tn = conf_mat[0, 0]
    fn = conf_mat[1, 0]

    return (tp / (tp + fn), fp / (fp + tn))


def conf_mat_to_f1(conf_mat):

    return float(2 * conf_mat[1, 1]) / float(2 * conf_mat[1, 1] +
                                             conf_mat[0, 1] + conf_mat[1, 0])


def getF1Score(gt_positives, seg_positives):

    f1 = np.zeros((maxIdx))

    for i in range(maxIdx):  #frames
        f1[i] = f1_score(gt_positives[i], seg_positives[i], average=None)

    return (f1)


def get_images(path, extension=('jpg', 'png')):
    """ Generates list of (sorted) images
    Returns List of paths to images
    """

    fnames = []

    if (isinstance(extension, str)):
        extension = [extension]

    for ext in extension:
        fnames += glob.glob(os.path.join(path, '*' + ext))

    fnames = sorted(fnames)

    return fnames


def makeFrameFileNames(frame_prefix,
                       frame_dir,
                       root_path,
                       ds_dir,
                       extension=('jpg', 'png'),
                       seqStart=None,
                       seqEnd=None):
    """ Generates list of (sorted) images
    Returns List of paths to images
    """

    fnames = []
    path = os.path.join(root_path, ds_dir, frame_dir)
    # path = root_path + ds_dir + '/' + frameDir + '/'

    if (isinstance(extension, str)):
        extension = [extension]

    for ext in extension:
        fnames.append(glob.glob(os.path.join(path, frame_prefix + '*' + ext)))

    fnames = [item for sublist in fnames for item in sublist]
    fnames = sorted(fnames)

    return fnames


def relabel(labels, map_dict):
    """ Relabel superpixel array from dictionary
    Returns array of labels
    """
    shape = labels.shape
    labels = labels.ravel()
    new_labels = np.copy(labels)
    for k, v in map_dict.items():
        new_labels[labels == k] = v

    return new_labels.reshape(shape)


def lp_sol_to_sps(g, tls, g_ss, x, labels=None):

    e = g.edges()
    e_marked = [e[i] for i in range(len(e)) if (x[i] > 0)]
    e_marked = [e for e in e_marked if ((e[0] != 's') & (e[1] != 't'))]
    all_ids = [e[0][0] for e in e_marked]

    sps = [tls[id_].sps[0] for id_ in all_ids]
    sps = [item for sublist in sps for item in sublist]
    #sps = [item for sublist in sps for item in sublist]

    sps_children = []
    for s in sps:
        this_frame = s[0]
        #print(s[1])
        if (type(s[1]) is np.int64):
            this_children = ss.get_children(g_ss[this_frame], s[1])
            for c in this_children:
                sps_children.append((this_frame, c[0]))
        if (type(s[1]) is np.ndarray):
            this_children = [
                ss.get_children(g_ss[this_frame], s[1][t])
                for t in range(s[1].shape[0])
            ]
            for c in this_children:
                sps_children.append((this_frame, c[0]))

    return np.asarray(sps_children)

    #sps = [t.sps[0][0] for t in tls if(t.id_ == )]


def norm_to_pix(x, y, width, height):

    j = int(np.round(x * (width - 1), 0))
    i = int(np.round(y * (height - 1), 0))

    return i, j


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def tracklet_set_to_sp_path(tls, set_):
    paths = list()

    # Get last set of KSP
    set_ = set_[-1]
    for s, i in zip(set_, range(len(set_))):
        # Get tracklets id_
        ids = [tup for tup in s if ((tup != 's') and (tup != 't'))]
        ids = ids[0::2]  # Keep entrance node
        ids = [n[0] for n in ids]
        p_ = list()
        for id_ in ids:
            p_.append([tl.sps for tl in tls if (tl.id_ == id_)][0])

        paths.append(np.concatenate(p_))

    return paths


def locs2d_to_sps(locs2d, labels):
    sps = list()
    # Convert input to marked (if necessary).

    for index, row in locs2d.iterrows():
        ci, cj = csv.coord2Pixel(row['x'], row['y'], labels.shape[1],
                                 labels.shape[0])
        sps.append((int(row['frame']), labels[ci, cj, int(row['frame'])]))

    return sps
