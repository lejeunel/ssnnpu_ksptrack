import my_utils as utls
import logging
import plot_results_ksp_simple as pksp
import os
import yaml

"""
Test self-learning on one sequence with one gaze set
"""

#dirs_ = ['Dataset01/results/2017-08-21_16-06-40_exp',
dir_ = ('/home/laurent.lejeune/medical-labeling/'
        'Dataset10/results/2017-09-27_14-28-45_exp')

with open(os.path.join(dir_, 'cfg.yml'), 'r') as outfile:
    conf = yaml.load(outfile)


utls.setup_logging(dir_)

logger = logging.getLogger('test_self_learning')

pksp.main(conf, logger=logger)
