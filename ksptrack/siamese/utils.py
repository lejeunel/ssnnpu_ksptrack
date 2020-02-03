import logging
import os
from os.path import join as pjoin
import yaml
import matplotlib.pyplot as plt
from skimage import segmentation
import numpy as np
import torch
import shutil
from tqdm import tqdm
import networkx as nx
from itertools import combinations


def prepare_couple_graphs(graphs, nn_radius):

    couple_graphs = []
    print('making NN-graphs')
    for i in tqdm(range(len(graphs) - 1)):
        mapping_1 = {n: n + graphs[i].number_of_nodes() for n in graphs[i+1]}
        g_1 = nx.relabel_nodes(graphs[i+1], mapping_1, copy=True)
        g = nx.compose(graphs[i], g_1)
        all_nodes = np.concatenate(([n for n in graphs[i].nodes()],
                                    [n for n in g_1.nodes()]))
        all_edges = np.array(list(combinations(all_nodes, 2)))

        centroids_x_0 = [graphs[i].nodes[n]['centroid'][0] for n in graphs[i].nodes()]
        centroids_x_1 = [g_1.nodes[n]['centroid'][0] for n in g_1.nodes()]
        centroids_x = np.concatenate((centroids_x_0, centroids_x_1), axis=0)
        centroids_y_0 = [graphs[i].nodes[n]['centroid'][1]
                         for n in graphs[i].nodes()]
        centroids_y_1 = [g_1.nodes[n]['centroid'][1] for n in g_1.nodes()]
        centroids_y = np.concatenate((centroids_y_0, centroids_y_1), axis=0)

        dists = np.sqrt((centroids_x[all_edges[:, 0]] - centroids_x[all_edges[:, 1]])**2 +
                        (centroids_y[all_edges[:, 0]] - centroids_y[all_edges[:, 1]])**2)
        inds = np.argwhere(dists < nn_radius).ravel()

        edges_nn = torch.from_numpy(all_edges[inds, :])
        couple_graphs.append(edges_nn)
    
    return couple_graphs



def batch_to_device(batch, device):

    return {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }


def to_onehot(arr_int, n_categories):
    b = np.zeros((arr_int.size, n_categories))
    b[np.arange(arr_int.size), arr_int] = 1
    return b



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
