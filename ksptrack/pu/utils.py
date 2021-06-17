import copy
import logging
import os
from itertools import combinations
from os.path import join as pjoin

import networkx as nx
import numpy as np
import torch
import yaml
from tqdm import tqdm


def df_to_tgt(df):
    target_pos = {
        r['frame']: torch.zeros(r['n_labels'])
        for _, r in df.iterrows()
    }

    for _, r in df.iterrows():
        if not np.isnan(r['label']):
            target_pos[r['frame']][int(r['label'])] = 1.

    frames = [r['frame'] for _, r in df.iterrows()]
    idx_f = np.unique(frames, return_index=True)[1]
    frames = [frames[idx] for idx in sorted(idx_f)]
    target_pos = torch.cat([target_pos[f] for f in frames])
    target_neg = torch.zeros_like(target_pos)
    target_neg[torch.logical_not(target_pos)] = 1.

    return target_pos, target_neg


def batch_to_device(batch, device):

    return {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }


def save_checkpoint(dict_, path):

    dir_ = os.path.split(path)[0]
    if (not os.path.exists(dir_)):
        os.makedirs(dir_)

    state_dict = dict_['model'].state_dict()

    torch.save(state_dict, path)
