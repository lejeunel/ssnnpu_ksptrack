import os
from os.path import join as pjoin
import yaml
import pandas as pd
import numpy as np
import ksptrack.graph_tracking as gtrack
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
from ksptrack.utils.link_agent import LinkAgent
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.link_agent_mask import LinkAgentMask
import logging
import ksptrack.sp_manager as spm
from ksptrack.utils import write_frames_results
from ksptrack.utils import comp_scores
import matplotlib.pyplot as plt
import datetime
from ksptrack.cfgs import params


def make_link_agent(labels, cfg):
    if(cfg.model_path is not None):
        return
    elif(cfg.entrance_masks_path is not None):
        return LinkAgentMask(csv_path=os.path.join(cfg.in_path, cfg.locs_dir,
                                                   cfg.csv_fname),
                             labels=labels,
                             thr_entrance=cfg.thr_entrance,
                             sigma=cfg.ml_sigma,
                             mask_path=cfg.entrance_masks_path)
    else:
        return LinkAgentRadius(csv_path=os.path.join(cfg.in_path, cfg.locs_dir,
                                                     cfg.csv_fname),
                               data_path=cfg.in_path,
                               thr_entrance=cfg.thr_entrance,
                               sigma=cfg.ml_sigma,
                               entrance_radius=cfg.norm_neighbor_in)


def main(cfg):
    data = dict()

    d = datetime.datetime.now()
    # cfg.run_dir = pjoin(cfg.out_path,
    #                     '{:%Y-%m-%d_%H-%M-%S}_{}'.format(d, cfg.exp_name))
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

    # Make frame file names
    cfg.frameFileNames = utls.get_images(
        os.path.join(cfg.in_path, cfg.frame_dir))

    locs2d = utls.readCsv(
        os.path.join(cfg.in_path, cfg.locs_dir, cfg.csv_fname))

    cfg.precomp_desc_path = pjoin(cfg.in_path, cfg.precomp_dir)
    if (not os.path.exists(cfg.precomp_desc_path)):
        os.makedirs(cfg.precomp_desc_path)

    with open(os.path.join(cfg.run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    cfg.gtFileNames = utls.get_images(os.path.join(cfg.in_path, cfg.truth_dir))

    # ---------- Descriptors/superpixel costs
    dm = DataManager(cfg)
    dm.calc_superpix()

    dm.calc_sp_feats(locs2d,
                     save_dir=cfg.precomp_desc_path,
                     mode=cfg.feats_mode)

    logger.info('Building superpixel managers')
    sps_man = spm.SuperpixelManager(dm,
                                    cfg,
                                    with_flow=cfg.use_hoof,
                                    init_mode=cfg.sp_trans_init_mode,
                                    init_radius=cfg.sp_trans_init_radius)

    link_agent = make_link_agent(dm.labels, cfg)

    locs2d_sps = link_agent.get_all_entrance_sps(dm.sp_desc_df)

    dm.calc_pm(np.array(locs2d_sps),
               all_feats_df=dm.sp_desc_df,
               mode='bagging')

    link_agent.update_trans_transform(dm.sp_desc_df,
                                      dm.fg_pm_df,
                                      [cfg.ml_down_thr, cfg.ml_up_thr],
                                      cfg.ml_n_samps,
                                      cfg.lfda_dim,
                                      cfg.lfda_k,
                                      embedding_type=cfg.ml_embedding)

    ksp_scores_mat = []

    g_for = gtrack.GraphTracking(link_agent, sps_man=sps_man)

    g_back = gtrack.GraphTracking(link_agent, sps_man=sps_man)

    find_new_forward = True
    find_new_backward = True
    i = 0

    pos_sp_for = []
    pos_sp_back = []
    list_ksp = []
    list_paths_for = []
    list_paths_back = []
    pos_tls_for = None
    pos_tls_back = None

    dict_ksp = dict()
    while ((find_new_forward or find_new_backward) and (i < cfg.n_iters_ksp)):

        logger.info("i: " + str(i + 1))

        if ((i > 0) & find_new_forward):
            g_for.merge_tracklets_temporally(pos_tls_for, dm.fg_pm_df,
                                             dm.sp_desc_df, dm.conf.pm_thr)

        if ((i > 0) & find_new_backward):
            g_back.merge_tracklets_temporally(pos_tls_back, dm.fg_pm_df,
                                              dm.sp_desc_df, dm.conf.pm_thr)
        # Make backward graph
        if (find_new_backward):

            g_back.makeFullGraph(dm.get_sp_desc_from_file(),
                                 dm.fg_pm_df,
                                 dm.conf.pm_thr,
                                 dm.conf.hoof_tau_u,
                                 direction='backward',
                                 labels=dm.labels)

            logger.info("Computing KSP on backward graph. (i: {}".format(i +
                                                                         1))
            g_back.run()
            dict_ksp['backward_sets'], pos_tls_back = utls.ksp2sps(
                g_back.kspSet, g_back.tracklets)
            dict_ksp['backward_tracklets'] = g_back.tracklets
            dict_ksp['backward_costs'] = g_back.costs

        # Make forward graph
        if (find_new_forward):

            g_for.makeFullGraph(dm.get_sp_desc_from_file(),
                                dm.fg_pm_df,
                                dm.conf.pm_thr,
                                dm.conf.hoof_tau_u,
                                direction='forward',
                                labels=dm.labels)

            logger.info("Computing KSP on forward graph. (i: {})".format(i +
                                                                         1))
            g_for.run()
            dict_ksp['forward_sets'], pos_tls_for = utls.ksp2sps(
                g_for.kspSet, g_for.tracklets)
            dict_ksp['forward_tracklets'] = g_for.tracklets

        if ((find_new_forward or find_new_backward)):

            ksp_scores_mat = utls.sp_tuples_to_mat(
                dict_ksp['forward_sets'] + dict_ksp['backward_sets'],
                dm.labels)

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

            fileOut = os.path.join(cfg.run_dir,
                                   'pm_scores_iter_{}.npz'.format(i))
            data = dict()
            data['ksp_scores_mat'] = ksp_scores_mat
            np.savez(fileOut, **data)

            # Recompute PM values
            if (i + 1 < cfg.n_iters_ksp):
                dm.calc_pm(pos_sps=np.array(
                    list(set(locs2d_sps + marked_back + marked_for))),
                           all_feats_df=dm.sp_desc_df,
                           mode='bagging')

            link_agent.update_trans_transform(dm.sp_desc_df,
                                              dm.fg_pm_df,
                                              [cfg.ml_down_thr, cfg.ml_up_thr],
                                              cfg.ml_n_samps,
                                              cfg.lfda_dim,
                                              cfg.lfda_k,
                                              embedding_type=cfg.ml_embedding)

            i += 1

    # Saving
    fileOut = os.path.join(cfg.run_dir, 'results.npz')
    data = dict()
    data['frameFileNames'] = cfg.frameFileNames
    data['n_iters_ksp'] = cfg.n_iters_ksp
    data['ksp_scores_mat'] = ksp_scores_mat
    data['pm_scores_mat'] = dm.get_pm_array()
    data['paths_back'] = dict_ksp['backward_sets']
    data['paths_for'] = dict_ksp['forward_sets']
    logger.info("Saving results and cfg to: " + fileOut)
    np.savez(fileOut, **data)

    g_for.save_all(os.path.join(cfg.run_dir, 'g_for'))
    g_back.save_all(os.path.join(cfg.run_dir, 'g_back'))

    logger.info("done")

    logger.info('Finished experiment: ' + cfg.run_dir)

    write_frames_results.main(cfg, cfg.run_dir, logger)
    comp_scores.main(cfg, cfg.run_dir, logger)

    return cfg


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-path', required=True)
    p.add('--in-path', required=True)
    p.add('--entrance-masks-path', default=None)
    p.add('--model-path', default=None)

    cfg = p.parse_args()

    main(cfg)
