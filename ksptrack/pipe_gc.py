from ksptrack.plots.entrance_paths import paths
from ksptrack.gc_optimize import optimize_params
from ksptrack.gc_refinement import main
from ksptrack.cfgs import params
from os.path import join as pjoin
import os
import numpy as np
import pandas as pd
import fnmatch

root_data_dir = '/home/ubelix/data/medical-labeling'
root_runs_dir = '/home/ubelix/runs/ksptrack'

# use cst_radius segmentation for parameter search
seqs = {'Dataset00': 15,
        'Dataset10': 52,
        'Dataset20': 13,
        'Dataset30': 16}

best_params = {'Dataset00': {'sigma': None, 'lambda': None, 'gamma': None},
               'Dataset10': {'sigma': None, 'lambda': None, 'gamma': None},
               'Dataset20': {'sigma': None, 'lambda': None, 'gamma': None},
               'Dataset30': {'sigma': None, 'lambda': None, 'gamma': None}}

gamma_range = np.arange(0.2,
                        0.6,
                        0.1)
lambda_range = np.arange(20,
                         60,
                         5)

for m in paths['cst_radius']:
    if(m[0] in seqs.keys()):
        path_csv = pjoin(root_runs_dir, m[0], m[1], 'best_gc_params.csv')
        if(not os.path.exists(path_csv)):
            g, l, s = optimize_params(pjoin(root_data_dir, m[0]),
                                    pjoin(root_runs_dir, m[0], m[1]),
                                    'input-frames',
                                    'ground_truth-frames',
                                    [seqs[m[0]]],
                                    gamma_range,
                                    lambda_range,
                                    sigma=None)
            best_params[m[0]]['gamma'] = g
            best_params[m[0]]['sigma'] = s
            best_params[m[0]]['lambda'] = l
        else:
            best_params[m[0]] = pd.read_csv(path_csv, index_col=0, header=None, squeeze=True)
    
# do all refinements
for m in paths.keys():
    for run in paths[m]:
        if(run[1] is not None):
            # get best params
            params = [v for k, v in best_params.items() if(fnmatch.fnmatch(k[:-1], m[0][:-1]+'*'))][0]
            main(pjoin(root_runs_dir, run[0], run[1]),
                params['gamma'],
                params['sigma'],
                params['lambda'],
                pjoin(root_data_dir, run[0]),
                'input-frames',
                'ground_truth-frames')
