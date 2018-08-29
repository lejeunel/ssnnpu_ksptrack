import os
from ruamel import yaml
from ksptrack.cfgs import cfg
import pandas as pd
import numpy as np
import ksptrack.graph_tracking as gtrack
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
import logging
import ksptrack.sp_manager as spm
from sklearn.metrics import (f1_score,
                             roc_curve)
import pickle
import matplotlib.pyplot as plt

def main(arg_cfg):
    data = dict()

    # Update config
    cfg_dict = cfg.cfg()
    arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['dataSetDir'])
    cfg_dict.update(arg_cfg)
    conf = cfg.dict_to_munch(cfg_dict)

    if not('dataOutDir' in arg_cfg.keys()):
        conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot,
                                            conf.dataSetDir,
                                            conf.resultDir,
                                            conf.fileOutPrefix,
                                            conf.testing)

    # Set logger
    utls.setup_logging(conf.dataOutDir)

    logger = logging.getLogger('ksp_'+conf.dataSetDir)

    logger.info('---------------------------')
    logger.info('starting experiment on: ' + conf.dataSetDir)
    logger.info('type of sequence: ' + conf.seq_type)
    logger.info('gaze filename: ' + conf.csvFileName_fg)
    logger.info('features type for graph: ' + conf.feats_graph)
    logger.info('Result dir:')
    logger.info(conf.dataOutDir)
    logger.info('---------------------------')

    # Make frame file names
    conf.frameFileNames = utls.makeFrameFileNames(
        conf.framePrefix, conf.frameDigits, conf.frameDir,
        conf.dataInRoot, conf.dataSetDir, conf.frameExtension)


    if(conf.csvFileType == 'pandas'):
        locs2d = pd.read_csv(os.path.join(conf.dataInRoot,
                                                  conf.dataSetDir,
                                                  conf.gazeDir,
                                                  conf.csvFileName_fg))
    elif(conf.csvFileType == 'anna'):
        locs2d = utls.readCsv(os.path.join(conf.dataInRoot,
                                                   conf.dataSetDir,
                                                   conf.gazeDir,
                                                   conf.csvFileName_fg))

    if (conf.labelMatPath != ''):
        conf.labelMatPath = os.path.join(conf.dataOutRoot,
                                         conf.dataSetDir,
                                         conf.frameDir,
                                         conf.labelMatPath)

    conf.precomp_desc_path = os.path.join(conf.dataOutRoot,
                                          conf.dataSetDir,
                                          conf.feats_files_dir)
    #conf.precomp_desc_path = conf.dataOutDir

    with open(os.path.join(conf.dataOutDir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(conf.__dict__,
                  stream=outfile,
                  default_flow_style=False)

    # ---------- Descriptors/superpixel costs
    my_dataset = DataManager(conf)
    if(conf.calc_superpix):
        my_dataset.calc_superpix(save=True)
    if(conf.calc_oflow):
        my_dataset.calc_oflow(save=True)

    my_dataset.load_superpix_from_file()
    if(conf.force_relabel):
        my_dataset.relabel(save=True)
    if(conf.calc_sp_feats_unet_rec):
        my_dataset.calc_sp_feats_unet_rec(save_dir=conf.precomp_desc_path)
    if(conf.calc_sp_feats_unet_gaze_rec):
        my_dataset.calc_sp_feats_unet_gaze_rec(locs2d,
                                               save_dir=conf.precomp_desc_path)
    if(conf.calc_sp_feats_vgg16):
        my_dataset.calc_sp_feats_vgg16(save_dir=conf.precomp_desc_path)

    logger.info('Building superpixel managers')
    sps_man = spm.SuperpixelManager(my_dataset,
                                    conf,
                                    with_flow=True,
                                    init_mode=conf.sp_trans_init_mode)

    if(conf.calc_pm):
        pd_csv = utls.pandas_to_std_csv(locs2d)
        my_dataset.calc_pm(pd_csv,
                           save=True,
                           marked_feats=None,
                           all_feats_df=my_dataset.get_sp_desc_from_file(),
                           in_type='csv_normalized',
                           max_n_feats=conf.max_n_feats,
                           mode='foreground',
                           feat_fields=['desc'],
                           n_jobs=conf.bagging_jobs)

    my_dataset.load_all_from_file()

    gt = None
    if(conf.monitor_score):
        logger.info('Making ground-truth seeds for monitoring')
        gtFileNames = utls.makeFrameFileNames(
            conf.framePrefix, conf.frameDigits, conf.gtFrameDir,
            conf.dataInRoot, conf.dataSetDir, conf.frameExtension)

        gt = utls.getPositives(gtFileNames)
        gt_seeds = utls.make_y_array_true(gt, my_dataset.labels)

    # Tracking with KSP---------------
    ksp_scores_mat = []

    #pm_arr = my_dataset.get_pm_array(frames=[0, 1, 2, 3])
    g_for = gtrack.GraphTracking(sps_man,
                                 tol=conf.ksp_tol,
                                 mode='edge')

    g_back = gtrack.GraphTracking(sps_man,
                                  tol=conf.ksp_tol,
                                  mode='edge')

    g_for.make_trans_transform(my_dataset.sp_desc_df,
                               my_dataset.fg_pm_df,
                               conf.thresh_aux_fix,
                               conf.lfda_n_samps,
                               conf.lfda_dim,
                               conf.lfda_k,
                               pca=conf.pca)

    g_back.make_trans_transform(my_dataset.sp_desc_df,
                                my_dataset.fg_pm_df,
                                conf.thresh_aux_fix,
                                conf.lfda_n_samps,
                                conf.lfda_dim,
                                conf.lfda_k,
                                conf.pca)

    find_new_forward = True
    find_new_backward = True
    i = 0

    pos_sp_for = []
    pos_sp_back = []
    list_ksp = []
    list_paths_for = []
    list_paths_back = []

    while((find_new_forward or find_new_backward) and (i < conf.n_iter_ksp)):
        dict_ksp = dict()

        logger.info("i: " + str(i+1))

        if(i == 0):
            # Make backward graph
            if(find_new_backward):

                g_back.makeFullGraph(
                    my_dataset.get_sp_desc_from_file(),
                    my_dataset.fg_pm_df,
                    my_dataset.centroids_loc,
                    utls.pandas_to_std_csv(locs2d),
                    my_dataset.conf.normNeighbor_in,
                    my_dataset.conf.thresh_aux_fix,
                    my_dataset.conf.tau_u,
                    direction='backward',
                    labels=my_dataset.labels)

            # Make forward graph
            if(find_new_forward):

                g_for.makeFullGraph(
                    my_dataset.get_sp_desc_from_file(),
                    my_dataset.fg_pm_df,
                    my_dataset.centroids_loc,
                    utls.pandas_to_std_csv(locs2d),
                    my_dataset.conf.normNeighbor_in,
                    my_dataset.conf.thresh_aux_fix,
                    my_dataset.conf.tau_u,
                    direction='forward',
                    labels=my_dataset.labels)

        else:
            g_for.merge_tracklets_temporally(pos_tls_for,
                                             my_dataset.centroids_loc,
                                             my_dataset.fg_pm_df,
                                             my_dataset.sp_desc_df,
                                             utls.pandas_to_std_csv(locs2d),
                                             my_dataset.conf.normNeighbor_in,
                                             my_dataset.conf.thresh_aux[-1],
                                             my_dataset.get_labels(),
                                             my_dataset.conf.tau_u)
            pd_csv = utls.pandas_to_std_csv(locs2d)

            g_back.merge_tracklets_temporally(pos_tls_back,
                                              my_dataset.centroids_loc,
                                              my_dataset.fg_pm_df,
                                              my_dataset.sp_desc_df,
                                              pd_csv,
                                              my_dataset.conf.normNeighbor_in,
                                              my_dataset.conf.thresh_aux[-1],
                                              my_dataset.get_labels(),
                                              my_dataset.conf.tau_u)

        logger.info("Computing KSP on forward graph. (i: {})".format(i+1))
        g_for.run()

        dict_ksp['forward_sets'], pos_tls_for = utls.ksp2sps(g_for.kspSet,
                                                             g_for.tracklets)
        dict_ksp['forward_tracklets'] = g_for.tracklets

        logger.info("Computing KSP on backward graph. (i: {}".format(i+1))
        g_back.run()
        dict_ksp['backward_sets'], pos_tls_back = utls.ksp2sps(g_back.kspSet,
                                                               g_back.tracklets)
        dict_ksp['backward_tracklets'] = g_back.tracklets
        dict_ksp['backward_costs'] = g_back.costs

        if((find_new_forward or find_new_backward)):

            ksp_scores_mat = utls.sp_tuples_to_mat(
                dict_ksp['forward_sets'] + dict_ksp['backward_sets'],
                my_dataset.get_labels())

            # Update marked superpixels if graph is not "finished"
            if(find_new_forward):
                this_marked_for =  utls.sps2marked(dict_ksp['forward_sets'])
                my_dataset.update_marked_sp(this_marked_for,
                                            mode='foreground')

                pos_sp_for.append(this_marked_for.shape[0])

                logger.info("""Forward graph. Number of positive sps
                            of ksp at iteration {}: {}""".format(i+1,
                                      this_marked_for.shape[0]))
                if(i > 0):
                    if(pos_sp_for[-1] <= pos_sp_for[-2]):
                        find_new_forward = False

            if(find_new_backward):
                this_marked_back =  utls.sps2marked(dict_ksp['backward_sets'])
                my_dataset.update_marked_sp(this_marked_back,
                                            mode='foreground')
                pos_sp_back.append(this_marked_back.shape[0])

                logger.info("""Backward graph. Number of positive sps of
                                ksp at iteration {}: {} """.format(i+1,
                                    this_marked_back.shape[0]))
                if(i > 0):
                    if(pos_sp_back[-1] <= pos_sp_back[-2]):
                        find_new_backward = False

            if(find_new_backward or find_new_forward):
                list_ksp.append(dict_ksp)

                n_pix_ksp = np.sum((ksp_scores_mat > 0).ravel())
                logger.info("""Number hit pixels of ksp at iteration {}:
                            {}""".format(i+1, n_pix_ksp))

                if(conf.monitor_score):
                    seeds = utls.make_full_y_from_ksp(dict_ksp,
                                                      my_dataset.sp_desc_df)
                    f1 = f1_score(gt_seeds[:, 2],seeds[:, 2])
                    fpr, tpr, _ = roc_curve(gt_seeds[:, 2], seeds[:, 2])

                    logger.info("""F1 (SP-level) at
                                iteration {}: {} """.format(i + 1, f1))
                    logger.info("""FPR/TPR (SP-level) at
                        iteration {}: {}/{} """.format(i+1, fpr[1], tpr[1]))

                fileOut = os.path.join(conf.dataOutDir,
                                       'pm_scores_iter_{}.npz'.format(i))
                data = dict()
                data['ksp_scores_mat'] = ksp_scores_mat
                np.savez(fileOut, **data)

                # Recompute PM values
                if(i+1 < conf.n_iter_ksp):
                    my_dataset.calc_pm(my_dataset.fg_marked,
                                       save=False,
                                       marked_feats=None,
                                       all_feats_df=my_dataset.sp_desc_df,
                                       in_type='not csv',
                                       mode='foreground',
                                       max_n_feats=conf.max_n_feats,
                                       max_depth=conf.max_depth,
                                       feat_fields=['desc'],
                                       max_samples=conf.max_samples,
                                       n_jobs=conf.bagging_jobs)

                    my_dataset.conf.thresh_aux.append(conf.thresh_aux_fix)

                    g_for.make_trans_transform(my_dataset.sp_desc_df,
                                               my_dataset.fg_pm_df,
                                               conf.lfda_thresh,
                                               conf.lfda_n_samps,
                                               conf.lfda_dim,
                                               conf.lfda_k,
                                               pca=conf.pca,
                                               n_comps_pca=conf.n_comp_pca)

                    g_back.make_trans_transform(my_dataset.sp_desc_df,
                                                my_dataset.fg_pm_df,
                                                conf.lfda_thresh,
                                                conf.lfda_n_samps,
                                                conf.lfda_dim,
                                                conf.lfda_k,
                                               pca=conf.pca,
                                               n_comps_pca=conf.n_comp_pca)

            i += 1

    # Saving KSP-------------------------------------------------
    fileOut = os.path.join(conf.dataOutDir, 'results.npz')
    data = dict()
    data['frameFileNames'] = conf.frameFileNames
    data['n_iter_ksp'] = conf.n_iter_ksp
    data['ksp_scores_mat'] = ksp_scores_mat
    #data['list_ksp'] = list_ksp
    data['list_paths_back'] = list_paths_back
    data['list_paths_for'] = list_paths_for
    logger.info("Saving results and config to: " + fileOut)
    np.savez(fileOut, **data)

    logger.info("done")


    logger.info('Finished experiment: ' + conf.dataOutDir)

    return conf, logger
