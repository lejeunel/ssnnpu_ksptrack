import logging
import os

import cfg
import dataset as ds
import yaml

import my_utils as utls


def main(arg_cfg):
    data = dict()

    #Update config
    cfg_dict = cfg.cfg()
    arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['ds_dir'])
    cfg_dict.update(arg_cfg)
    conf = cfg.Bunch(cfg_dict)

    #Write config to result dir
    conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot, conf.ds_dir,
                                         conf.resultDir, conf.out_dir_prefix,
                                         conf.testing)

    #Set logger
    utls.setup_logging(conf.dataOutDir)

    logger = logging.getLogger('feat_extr')


    logger.info('---------------------------')
    logger.info('starting feature extraction on: ' + conf.ds_dir)
    logger.info('type of sequence: ' + conf.seq_type)
    logger.info('gaze filename: ' + conf.csvFileName_fg)
    logger.info('features type: ' + conf.feat_extr_algorithm)
    logger.info('Result dir:')
    logger.info(conf.dataOutDir)
    logger.info('---------------------------')

    #Make frame file names
    gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
    gtFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.truth_dir,
        conf.root_path, conf.ds_dir, conf.frame_extension)

    conf.frameFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.frame_dir,
        conf.root_path, conf.ds_dir, conf.frame_extension)

    conf.myGaze_fg = utls.readCsv(os.path.join(conf.root_path,conf.ds_dir,conf.locs_dir,conf.csvFileName_fg))

    #conf.myGaze_bg = utls.readCsv(conf.csvName_bg)
    gt_positives = utls.getPositives(gtFileNames)

    if (conf.labelMatPath != ''):
        conf.labelMatPath = os.path.join(conf.dataOutRoot, conf.ds_dir, conf.frameDir,
                                    conf.labelMatPath)

    conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                    conf.feats_dir)

    # ---------- Descriptors/superpixel costs
    my_dataset = ds.Dataset(conf)

    my_dataset.load_superpix_from_file()
    my_dataset.calc_sp_feats_unet_gaze_rec(save=True)

    with open(os.path.join(conf.dataOutDir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=True)

    logger.info('Finished feature extraction: ' + conf.ds_dir)

    return conf
