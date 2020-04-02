import os
from os.path import join as pjoin
import yaml
import numpy as np
import ksptrack.graph_tracking as gtrack
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.link_agent_model import LinkAgentModel
import logging
import ksptrack.sp_manager as spm
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese.modeling.dec import DEC
from ksptrack.utils import write_frames_results
from ksptrack.utils import comp_scores
import datetime
from ksptrack.cfgs import params
import torch
import pandas as pd


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
    sp_desc_df['positive'] = (sp_desc_df['positive_x'] | sp_desc_df['positive_y'])
    sp_desc_df = sp_desc_df.drop(columns=['positive_x', 'positive_y'])
    return sp_desc_df


def make_link_agent(cfg):

    if (cfg.siam_path):
        cfg_path = os.path.split(cfg.siam_path)[0]
        cfg_path = os.path.split(cfg_path)[0]
        cfg_path = pjoin(cfg_path, 'cfg.yml')
        with open(cfg_path) as f:
            cfg_siam = Bunch(yaml.load(f, Loader=yaml.FullLoader))
        model = Siamese(cfg_siam.embedded_dims, cfg_siam.n_clusters,
                        cfg_siam.alpha,
                        cfg_siam.backbone)
        print('loading checkpoint {}'.format(cfg.siam_path))
        state_dict = torch.load(cfg.siam_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=True)
        link_agent = LinkAgentModel(csv_path=pjoin(cfg.in_path, cfg.locs_dir,
                                                   cfg.csv_fname),
                                    data_path=cfg.in_path,
                                    model=model,
                                    entrance_radius=cfg.norm_neighbor_in,
                                    cuda=cfg.cuda)
        # compute features
        path_feats = pjoin(cfg.in_path, cfg.precomp_dir, 'sp_desc_siam.p')
        path_centroids = pjoin(cfg.in_path, cfg.precomp_dir,
                               'centroids_loc_df.p')

        print('computing features to {}'.format(path_feats))
        feats = [item for sublist in link_agent.feats for item in sublist]
        centroids = pd.read_pickle(path_centroids)
        feats = centroids.assign(desc=feats)
        print('Saving features to {}'.format(path_feats))
        feats.to_pickle(path_feats)

        return link_agent, feats


def main(cfg):

    data = dict()

    d = datetime.datetime.now()
    cfg.run_dir = pjoin(cfg.out_path, '{}'.format(cfg.exp_name))

    if (os.path.exists(cfg.run_dir)):
        print('run dir {} already exists.'.format(cfg.run_dir))
        return cfg
    else:
        os.makedirs(cfg.run_dir)

    # Set logger
    utls.setup_logging(cfg.run_dir)
    logger = logging.getLogger('ksp')

    logger.info('-' * 10)
    logger.info('starting experiment on: {}'.format(cfg.in_path))
    logger.info('2d locs filename: {}'.format(cfg.csv_fname))
    logger.info('Output path: {}'.format(cfg.run_dir))
    logger.info('-' * 10)

    precomp_desc_path = pjoin(cfg.in_path, cfg.precomp_dir)
    if (not os.path.exists(precomp_desc_path)):
        os.makedirs(precomp_desc_path)

    with open(pjoin(cfg.run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    # ---------- Descriptors/superpixel costs
    dm = DataManager(cfg.in_path, cfg.precomp_dir, feats_mode=cfg.feats_mode)
    dm.calc_superpix(cfg.slic_compactness, cfg.slic_n_sp)

    link_agent, desc_df = make_link_agent(cfg)

    logger.info('Building superpixel managers')
    sps_man = spm.SuperpixelManager(cfg.in_path,
                                    cfg.precomp_dir,
                                    link_agent.labels,
                                    desc_df,
                                    init_radius=cfg.sp_trans_init_radius,
                                    hoof_n_bins=cfg.hoof_n_bins)


    locs2d_sps = link_agent.get_all_entrance_sps(desc_df)
    desc_df['positive'] = locs2d_sps

    if(cfg.use_siam_pred):
        pm = utls.probas_to_df(link_agent.labels, link_agent.obj_preds)
    else:
        pm = utls.calc_pm(desc_df,
                          desc_df['positive'], cfg.bag_n_feats, cfg.bag_t,
                          cfg.bag_max_depth, cfg.bag_max_samples, cfg.bag_jobs)

    ksp_scores_mat = []

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
    while ((find_new_forward or find_new_backward) and (i < cfg.n_iters_ksp)):

        logger.info("i: " + str(i + 1))

        if ((i > 0) & find_new_forward):
            g_for.merge_tracklets_temporally(pos_tls_for, pm,
                                             desc_df, cfg.pm_thr)

        if ((i > 0) & find_new_backward):
            g_back.merge_tracklets_temporally(pos_tls_back, pm,
                                              desc_df, cfg.pm_thr)
        # Make backward graph
        if (find_new_backward):

            g_back.makeFullGraph(desc_df,
                                 pm,
                                 cfg.pm_thr,
                                 cfg.hoof_tau_u,
                                 cfg.norm_neighbor,
                                 direction='backward',
                                 labels=link_agent.labels)

            logger.info("Computing KSP on backward graph. (i: {}".format(i +
                                                                         1))
            g_back.run()
            dict_ksp['backward_sets'], pos_tls_back = utls.ksp2sps(
                g_back.kspSet, g_back.tracklets)
            dict_ksp['backward_tracklets'] = g_back.tracklets
            dict_ksp['backward_costs'] = g_back.costs

        # Make forward graph
        if (find_new_forward):

            g_for.makeFullGraph(desc_df,
                                pm,
                                cfg.pm_thr,
                                cfg.hoof_tau_u,
                                cfg.norm_neighbor,
                                direction='forward',
                                labels=link_agent.labels)

            logger.info("Computing KSP on forward graph. (i: {})".format(i +
                                                                         1))
            g_for.run()
            dict_ksp['forward_sets'], pos_tls_for = utls.ksp2sps(
                g_for.kspSet, g_for.tracklets)
            dict_ksp['forward_tracklets'] = g_for.tracklets

        if ((find_new_forward or find_new_backward)):

            ksp_scores_mat = utls.sp_tuples_to_mat(
                dict_ksp['forward_sets'] + dict_ksp['backward_sets'],
                link_agent.labels)

            # Update marked superpixels if graph is not "finished"
            if (find_new_forward):
                marked_for = [
                    m for sublist in dict_ksp['forward_sets'] for m in sublist
                ]
                pos_sp_for.append(len(marked_for))

                logger.info("""Forward graph. Number of positive sps
                            of ksp at iteration {}: {}""".format(
                    i + 1, len(marked_for)))
                if (i > 0):
                    if (pos_sp_for[-1] <= pos_sp_for[-2]):
                        find_new_forward = False

            if (find_new_backward):
                marked_back = [
                    m for sublist in dict_ksp['backward_sets'] for m in sublist
                ]
                pos_sp_back.append(len(marked_back))

                logger.info("""Backward graph. Number of positive sps of
                                ksp at iteration {}: {} """.format(
                    i + 1, len(marked_back)))
                if (i > 0):
                    if (pos_sp_back[-1] <= pos_sp_back[-2]):
                        find_new_backward = False

            list_ksp.append(dict_ksp)

            n_pix_ksp = np.sum((ksp_scores_mat > 0).ravel())
            logger.info("""Number hit pixels of ksp at iteration {}:
                        {}""".format(i + 1, n_pix_ksp))

            fileOut = pjoin(cfg.run_dir, 'pm_scores_iter_{}.npz'.format(i))
            data = dict()
            data['ksp_scores_mat'] = ksp_scores_mat
            np.savez(fileOut, **data)

            # Recompute PM values
            if (i + 1 < cfg.n_iters_ksp):
                desc_df = merge_positives(desc_df,
                                          marked_for,
                                          marked_back)
                pm = utls.calc_pm(
                    desc_df,
                    desc_df['positive'],
                    cfg.bag_n_feats, cfg.bag_t, cfg.bag_max_depth,
                    cfg.bag_max_samples, cfg.bag_jobs)

            i += 1

    # Saving
    fileOut = pjoin(cfg.run_dir, 'results.npz')
    data = dict()
    data['n_iters_ksp'] = cfg.n_iters_ksp
    data['ksp_scores_mat'] = ksp_scores_mat
    data['pm_scores_mat'] = utls.get_pm_array(link_agent.labels, pm)
    data['paths_back'] = dict_ksp['backward_sets']
    data['paths_for'] = dict_ksp['forward_sets']
    logger.info("Saving results and cfg to: " + fileOut)
    np.savez(fileOut, **data)

    logger.info("done")

    logger.info('Finished experiment: ' + cfg.run_dir)

    write_frames_results.main(cfg, cfg.run_dir, logger)
    comp_scores.main(cfg)

    return cfg


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-path', required=True)
    p.add('--in-path', required=True)
    p.add('--siam-path', default='')
    p.add('--use-siam-pred', default=False, action='store_true')

    cfg = p.parse_args()

    main(cfg)
