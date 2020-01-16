import logging
import os
from os.path import join as pjoin
import yaml
import matplotlib.pyplot as plt
from skimage import segmentation
import numpy as np
import torch
import shutil


def batch_to_device(batch, device):

    return {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }


def to_onehot(arr_int, n_categories):
    b = np.zeros((arr_int.size, n_categories))
    b[np.arange(arr_int.size), arr_int] = 1
    return b


def sample_edges(g, n_edges):

    pos_edges = [(n0, n1) for n0, n1 in g.edges()
                 if (g.edges[(n0, n1)]['clust_sim'] == 1)]
    pos_edges_conf = np.array(
        [g.edges[(n0, n1)]['avg_conf'] for n0, n1 in pos_edges])
    pos_edges_conf = pos_edges_conf / pos_edges_conf.sum()
    neg_edges = [(n0, n1) for n0, n1 in g.edges()
                 if (g.edges[(n0, n1)]['clust_sim'] == 0)]
    neg_edges_conf = np.array(
        [g.edges[(n0, n1)]['avg_conf'] for n0, n1 in neg_edges])
    neg_edges_conf = neg_edges_conf / neg_edges_conf.sum()

    pos_idx = np.random.choice(len(pos_edges), size=n_edges, p=pos_edges_conf)
    pos_edges = [pos_edges[i] for i in pos_idx]

    neg_idx = np.random.choice(len(neg_edges), size=n_edges, p=neg_edges_conf)
    neg_edges = [neg_edges[i] for i in neg_idx]

    idxs = np.concatenate((pos_edges, neg_edges))
    y = np.concatenate((np.ones(n_edges), np.zeros(n_edges)))

    return idxs, y


def sample_batch(graphs, clusters, feats, n_edges):

    Y = []
    X = []

    n_labels = [g.number_of_nodes() for g in graphs]
    if (isinstance(clusters, torch.Tensor)):
        clusters = torch.split(clusters, n_labels, dim=0)

    if (isinstance(feats, torch.Tensor)):
        feats = torch.split(feats, n_labels, dim=0)

    # set cluster indices and cluster similarities to graphs
    for g, clusters, feats in zip(graphs, clusters, feats):
        node_list = [(n,
                      dict(cluster=torch.argmax(c).cpu().item(),
                           confidence=torch.max(c).cpu().item()))
                     for n, c in zip(g.nodes(), clusters)]
        g.add_nodes_from(node_list)
        edge_list = [
            (n0, n1,
             dict(clust_sim=0 if
                  (g.nodes[n0]['cluster'] != g.nodes[n1]['cluster']) else 1,
                  avg_conf=np.mean(
                      (g.nodes[n0]['confidence'], g.nodes[n1]['confidence']))))
            for n0, n1 in g.edges()
        ]
        g.add_edges_from(edge_list)

        idx_nodes, y = sample_edges(g, n_edges)
        Y.append(torch.tensor(y))

        X_ = torch.stack([
            torch.stack((feats[n0], feats[n1]), dim=0) for n0, n1 in idx_nodes
        ],
                         dim=1)
        X.append(X_)

    X = torch.stack(X)
    Y = torch.stack(Y)

    return X, Y


def setup_logging(log_path,
                  conf_path='logging.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = conf_path

    # Get absolute path to logging.yaml
    path = pjoin(os.path.dirname(__file__), path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            config['handlers']['info_file_handler']['filename'] = pjoin(
                log_path, 'info.log')
            config['handlers']['error_file_handler']['filename'] = pjoin(
                log_path, 'error.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)



def save_checkpoint(dict_,
                    is_best,
                    path,
                    fname_cp='checkpoint.pth.tar',
                    fname_bm='best_model.pth.tar'):

    cp_path = os.path.join(path, fname_cp)
    bm_path = os.path.join(path, fname_bm)

    if (not os.path.exists(path)):
        os.makedirs(path)

    try:
        state_dict = dict_['model'].module.state_dict()
    except AttributeError:
        state_dict = dict_['model'].state_dict()

    torch.save(state_dict, cp_path)

    if (is_best):
        shutil.copyfile(cp_path, bm_path)
