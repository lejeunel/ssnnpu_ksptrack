import os
import iterative_ksp
import test_trans_costs
import test_cv_flow
import numpy as np
import datetime
import cfg_unet
import yaml
import numpy as np
import matplotlib.pyplot as plt
import results_dirs as rd
import plot_results_ksp_simple as pksp
import my_utils as utls
import logging

for key in rd.confs_dict_ksp.keys():
    for dset in range(len(rd.confs_dict_ksp[key])):
        for gset in range(len(rd.confs_dict_ksp[key][dset])):
            conf = rd.confs_dict_ksp[key][dset][gset]
            utls.setup_logging(conf.dataOutDir)
            logger = logging.getLogger('ksp_scoring_'+conf.dataSetDir)
            pksp.main(conf, logger=logger)
