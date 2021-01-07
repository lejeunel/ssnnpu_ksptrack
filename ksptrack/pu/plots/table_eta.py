#!/usr/bin/env python3

from os.path import join as pjoin
import glob
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tf2pd import tflog2pandas
from ksptrack.pu.plots.sp_errors import get_df as get_df_sp_errors
from argparse import Namespace
from ksptrack.pu.plots.table_results import parse_ksp_exps, myformat
import re
import configargparse


def add_level_eta(df, in_field='Methods', out_field='eta', mod_in_field=''):
    df[out_field] = 0.

    # search for number in string (assumed unique)
    for i, r in df.iterrows():
        field = r[in_field]
        eta = float(re.findall("\d+\.\d+", field)[0])
        df.loc[i, out_field] = eta
        if mod_in_field:
            df.loc[i, in_field] = mod_in_field

    return df


type_to_ind = {'Tweezer': 0, 'Cochlea': 1, 'Slitlamp': 2, 'Brain': 3}

if __name__ == "__main__":

    p = configargparse.ArgParser()
    p.add('--types',
          nargs='+',
          default=['Tweezer', 'Slitlamp', 'Brain', 'Cochlea'])
    p.add('--root-path', default='/home/ubelix/lejeune/runs/ksptrack')
    p.add('--save-path', default='table_eta.tex')

    cfg = p.parse_args()

    exp_names = {
        'nnpuss': 'pu_piovrs',
        'nnputrue': 'pu_true',
        'nnpuconst': 'pu_meantrue_cst_piovrs'
    }

    order = {
        'nnputrue': 'KSPTrack/nnPUtrue',
        'nnpuss': 'KSPTrack/nnPUss',
        'nnpuconst': 'KSPTrack/nnPUconst',
    }

    eta = np.round(np.linspace(0.8, 1.8, 6), decimals=1)

    df_ss = parse_ksp_exps(
        cfg.root_path, cfg.types,
        ['{}_{:.1f}_ph2'.format(exp_names['nnpuss'], eta_) for eta_ in eta])
    df_ss = add_level_eta(df_ss, mod_in_field=order['nnpuss'])

    df_true = parse_ksp_exps(cfg.root_path, cfg.types, [exp_names['nnputrue']])
    df_const = parse_ksp_exps(cfg.root_path, cfg.types, [
        '{}_{:.1f}'.format(exp_names['nnpuconst'], eta_)
        for eta_ in eta if eta_ <= 1.4
    ])
    df_const = add_level_eta(df_const)
    df_const = add_level_eta(df_const, mod_in_field=order['nnpuconst'])

    df_true['Methods'] = order['nnputrue']

    df_eta = pd.concat((df_ss, df_const), axis=0)

    # df_eta = df_eta.drop(columns=['PR', 'RC'])

    df_mean = df_eta.groupby(['Types', 'Methods', 'eta']).mean()
    df_std = df_eta.groupby(['Types', 'Methods',
                             'eta']).std().rename(columns={
                                 'F1': '_F1',
                                 'PR': '_PR',
                                 'RC': '_RC'
                             })
    df_eta = pd.concat((df_mean, df_std), axis=1)

    # df_true = df_true.drop(columns=['PR', 'RC'])
    df_mean = df_true.groupby(['Types', 'Methods']).mean()
    df_std = df_true.groupby(['Types', 'Methods']).std().rename(columns={
        'F1': '_F1',
        'PR': '_PR',
        'RC': '_RC'
    })

    df_true = pd.concat((df_mean, df_std), axis=1)
    df_true['eta'] = '-'
    df_true.set_index('eta', append=True, inplace=True)

    df_all = pd.concat((df_eta, df_true)).sort_index()
    df_all = df_all.round(decimals=2)

    # add bold on ss and const
    # df_all['bold'] = False
    # for t in types:
    #     df_ = df_all.loc[t].loc[slice('KSPTrack/nnPUss', 'KSPTrack/nnPUconst')]
    #     idx = df_['F1'].values.argmax()
    #     df_.iloc[idx]['bold'] = True

    df_all = df_all.groupby(['Types', 'Methods']).apply(myformat)
    df_all = df_all.drop(columns=['_F1', '_PR', '_RC'])
    names = list(df_all.index.names)
    names[-1] = '$\eta$'
    df_all.index.set_names(names, inplace=True)

    caption = """
    Quantitative results on all datasets for different prior levels. We report the F1 score, precision (PR) and recall(RC) and standard deviations.
    """

    print('writing table to {}'.format(cfg.save_path))
    table = df_all.to_latex(escape=False,
                            column_format='llp{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}',
                            multirow=True,
                            caption=caption,
                            label='tab:results_eta')

    n_cols = 6
    # add horiz line below ours
    with open(cfg.save_path, 'w') as tf:
        for line in table.splitlines():
            # if 'KSPTrack/nnPUtrue' in line:
            #     line += '\n\hdashline{2-' + str(n_cols) + '}'
            # elif 'KSPTrack/nnPUconst' in line:
            #     line += '\n\hdashline{2-' + str(n_cols) + '}'
            if line.startswith('\\begin{table}'):
                line = '\\begin{table*}'
            elif line.startswith('\\end{table}'):
                line = '\\end{table*}'
            tf.write(line + '\n')
