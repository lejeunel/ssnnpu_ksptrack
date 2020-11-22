#!/usr/bin/env python3

from os.path import join as pjoin
import glob
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from ksptrack.exps import results_dirs as rd
import numpy as np
from tf2pd import tflog2pandas
from ksptrack.pu.plots.table_results import myformat, parse_ksp_exps

if __name__ == "__main__":

    types = ['Tweezer', 'Cochlea', 'Slitlamp', 'Brain']
    ksp_root_path = pjoin('/home/ubelix/lejeune/runs/ksptrack')
    model_root_path = pjoin('/home/ubelix/lejeune/runs/siamese_dec')
    out_path = '/home/laurent/Documents/papers/paper_pu/tables/results_vs_true.tex'

    piovrs = '1.6'
    ur = '0.12'

    exp_names = {
        'true': 'pu_pr_true_ph2',
        'nnpualt': 'pu_piovrs_{}_ph2'.format(piovrs)
    }

    order = {
        'ksp/' + exp_names['true']: 'KSPTrack/nnPUtrue',
        'ksp/' + exp_names['nnpualt']: 'KSPTrack/nnPU'
    }

    df = parse_ksp_exps(ksp_root_path, types, exp_names.values())
    df_mean = df.groupby(['Types', 'Methods']).mean()
    df_std = df.groupby(['Types', 'Methods']).std().rename(columns={
        'F1': '_F1',
        'PR': '_PR',
        'RC': '_RC'
    })
    df = pd.concat((df_mean, df_std), axis=1)

    # build full table
    df = df.round(decimals=2)

    # rename
    df = df.rename(index=order)

    # add bold field
    df['bold'] = False
    for t in types:
        idx = df.loc[t]['F1'].values.argmax()
        df.loc[t]['bold'].iloc[idx] = True

    # compute mean over all types
    means = df.groupby(['Methods'])['F1'].mean()
    std = df.groupby(['Methods'])['_F1'].mean()

    df = df.groupby(['Types', 'Methods']).apply(myformat)

    # remove dummy columns
    df = df.drop(columns=['_F1', '_PR', '_RC', 'bold'])

    df = df.drop(columns=['PR', 'RC'])
    df = df.reset_index().pivot('Methods', 'Types')
    df = df.reindex(order.values())

    # take only F1
    df = df[['F1']]

    df.columns = df.columns.droplevel()
    print(df)

    caption = """
    Quantitative results on all datasets of proposed method with true class-priors. We report the F1 scores and standard deviations.
    """

    print('writing table to {}'.format(out_path))
    table = df.to_latex(
        escape=False,
        column_format='llp{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}',
        multirow=True,
        caption=caption,
        label='tab:results_vs_true')

    # add horiz line below ours
    with open(out_path, 'w') as tf:
        for line in table.splitlines():
            if line.startswith('\\begin{table}'):
                line = '\\begin{table*}[t]'
            elif line.startswith('\\end{table}'):
                line = '\\end{table*}'
            tf.write(line + '\n')
