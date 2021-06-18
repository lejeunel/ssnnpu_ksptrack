import datetime
import logging
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
import yaml

import ksptrack.tracking.graph_tracking as gtrack
import ksptrack.utils.sp_manager as spm
from ksptrack import params
from ksptrack.utils import comp_scores
from ksptrack.utils import my_utils as utls
from ksptrack.utils import write_frames_results
from ksptrack.utils.superpixel_extractor import SuperpixelExtractor
from ksptrack.utils.link_agent_radius import LinkAgentRadius


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def merge_positives(sp_desc_df, pos_for, pos_back):

    to_add = pd.DataFrame(list(set(pos_back + pos_for)),
                          columns=['frame', 'label'])
    to_add['positive'] = True
    sp_desc_df = pd.merge(sp_desc_df,
                          to_add,
                          how='left',
                          on=['frame', 'label']).fillna(False)
    sp_desc_df['positive'] = (sp_desc_df['positive_x']
                              | sp_desc_df['positive_y'])
    sp_desc_df = sp_desc_df.drop(columns=['positive_x', 'positive_y'])
    return sp_desc_df


def make_link_agent(cfg):

    link_agent = LinkAgentRadius(csv_path=pjoin(cfg.in_path, cfg.locs_dir,
                                                cfg.locs_fname),
                                 data_path=cfg.in_path,
                                 entrance_radius=cfg.norm_neighbor_in,
                                 sp_labels_fname=cfg.sp_labels_fname,
                                 model_pred_path=cfg.model_path,
                                 cuda=cfg.cuda)

    # compute features
    path_feats = pjoin(cfg.in_path, cfg.precomp_dir, 'sp_desc.p')

    print('computing features to {}'.format(path_feats))
    positions = link_agent.pos
    positive = link_agent.labels_pos
    rows = [{
        'frame': f,
        'label': l,
        'x': positions[f][l, 0],
        'y': positions[f][l, 1],
        'positive': positive[f][l],
    } for f in range(len(positions)) for l in range(len(positions[f]))]
    print('Saving features to {}'.format(path_feats))
    df = pd.DataFrame(rows)
    df.to_pickle(path_feats)

    return link_agent, df


def main(cfg):

    data = dict()
    cfg.run_dir = pjoin(cfg.out_path, '{}'.format(cfg.exp_name))

    if (os.path.exists(cfg.run_dir)):
        print('run dir {} already exists.'.format(cfg.run_dir))
        # return cfg
    else:
        os.makedirs(cfg.run_dir)

    # Set logger
    print('-' * 10)
    print('starting experiment on: {}'.format(cfg.in_path))
    print('2d locs filename: {}'.format(cfg.locs_fname))
    print('Output path: {}'.format(cfg.run_dir))
    print('-' * 10)

    precomp_desc_path = pjoin(cfg.in_path, cfg.precomp_dir)
    if (not os.path.exists(precomp_desc_path)):
        os.makedirs(precomp_desc_path)

    with open(pjoin(cfg.run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    # ---------- Descriptors/superpixel costs
    spext = SuperpixelExtractor(cfg.in_path, cfg.precomp_dir,
                                cfg.slic_compactness, cfg.slic_n_sp)
    spext.run()

    link_agent, desc_df = make_link_agent(cfg)

    print('Building superpixel managers')
    sps_man = spm.SuperpixelManager(cfg.in_path, cfg.precomp_dir,
                                    link_agent.labels, desc_df,
                                    cfg.init_radius)
    print('Using foreground model from model')
    pm = utls.probas_to_df(link_agent.labels, link_agent.obj_preds)

    g_for = gtrack.GraphTracking(link_agent, sps_man=sps_man)

    g_back = gtrack.GraphTracking(link_agent, sps_man=sps_man)

    find_new_forward = True
    find_new_backward = True
    i = 0

    pos_sp_for = []
    pos_sp_back = []
    list_ksp = []
    pos_tls_for = None
    pos_tls_back = None

    dict_ksp = dict()

    # make forward and backward graphs
    g_back.make_graph(desc_df,
                      pm,
                      cfg.pm_thr,
                      cfg.norm_neighbor,
                      direction='backward',
                      labels=link_agent.labels)
    g_for.make_graph(desc_df,
                     pm,
                     cfg.pm_thr,
                     cfg.norm_neighbor,
                     direction='forward',
                     labels=link_agent.labels)

    print("Computing KSP on backward graph.")
    sps = g_back.run()
    dict_ksp['backward_sets'] = sps

    print("Computing KSP on forward graph.")
    sps = g_for.run()
    dict_ksp['forward_sets'] = sps

    all_sps = list(set(dict_ksp['forward_sets'] + dict_ksp['backward_sets']))
    print('got ', len(all_sps), ' unique superpixels')

    # Saving
    fileOut = pjoin(cfg.run_dir, 'results.npz')
    data = dict()
    data['ksp_scores_mat'] = utls.get_binary_array(link_agent.labels,
                                                   np.array(all_sps))
    data['pm_scores_mat'] = utls.get_pm_array(link_agent.labels, pm)
    data['paths_back'] = dict_ksp['backward_sets']
    data['paths_for'] = dict_ksp['forward_sets']
    data['all_sps'] = all_sps
    print("Saving results and cfg to: " + fileOut)
    np.savez(fileOut, **data)

    print('Finished experiment: ' + cfg.run_dir)

    write_frames_results.main(cfg, cfg.run_dir)
    comp_scores.main(cfg)

    return cfg


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-path', required=True)
    p.add('--in-path', required=True)
    p.add('--pred-path', default='')
    p.add('--trans-path', default='')
    p.add('--use-model-pred', default=False, action='store_true')
    # p.add('--loc-prior', default=False, action='store_true')
    p.add('--coordconv', default=False, action='store_true')
    p.add('--trans', default='lfda', type=str)

    cfg = p.parse_args()

    main(cfg)
