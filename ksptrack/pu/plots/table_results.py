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
import configargparse


# add bold tags
def myformat(r):
    if 'bold' in r:
        if (r['bold'].iat[0]):
            r['F1'] = '$\\bm{' + r['F1'].apply(
                str) + '} \pm ' + r['_F1'].apply(str) + '$'
        else:
            r['F1'] = '$' + r['F1'].apply(str) + ' \pm ' + r['_F1'].apply(
                str) + '$'
    else:
        r['F1'] = '$' + r['F1'].apply(str) + ' \pm ' + r['_F1'].apply(
            str) + '$'

    if 'PR' in r:
        r['PR'] = '$' + r['PR'].apply(str) + ' \pm ' + r['_PR'].apply(
            str) + '$'
    if 'RC' in r:
        r['RC'] = '$' + r['RC'].apply(str) + ' \pm ' + r['_RC'].apply(
            str) + '$'

    return r


type_to_ind = {'Tweezer': 0, 'Cochlea': 1, 'Slitlamp': 2, 'Brain': 3}


def parse_ksp_exps(root_path, types, exp_names):
    records = []

    max_seqs = 4

    for i, t in enumerate(types):
        dset_paths = sorted(
            glob.glob(pjoin(root_path, 'Dataset' + str(type_to_ind[t]) + '*')))
        for dset_path in dset_paths[:max_seqs]:
            dset_dir = os.path.split(dset_path)[-1]
            exp_paths = [pjoin(dset_path, x) for x in exp_names]
            for exp_path in exp_paths:
                score_path = pjoin(exp_path, 'scores.csv')
                if (os.path.exists(score_path)):
                    df = pd.read_csv(score_path,
                                     index_col=0,
                                     header=None,
                                     squeeze=True)
                    with open(pjoin(exp_path, 'cfg.yml')) as f:
                        cfg = yaml.load(f, Loader=yaml.FullLoader)

                    records.append([
                        t, dset_dir, 'ksp/' + os.path.split(exp_path)[-1],
                        df['f1_ksp'], df['pr_ksp'], df['rc_ksp']
                    ])
                else:
                    print(score_path, ' not found')

    df = pd.DataFrame.from_records(records,
                                   columns=('Types', 'dset', 'Methods', 'F1',
                                            'PR', 'RC'))
    return df


def parse_model_exps(root_path, types, exp_names, metrics=['F1']):
    records = []

    max_seqs = 4

    for i, t in enumerate(types):
        dset_paths = sorted(
            glob.glob(pjoin(root_path, 'Dataset' + str(i) + '*')))
        for dset_path in dset_paths[:max_seqs]:
            dset_dir = os.path.split(dset_path)[-1]
            exp_paths = [pjoin(dset_path, x) for x in exp_names]
            for exp_path in exp_paths:
                event_path = glob.glob(pjoin(exp_path, 'event*'))
                if len(event_path) > 0:
                    event_path = event_path[0]
                    # print(event_path)
                    if (os.path.exists(event_path)):
                        df = tflog2pandas(event_path)
                        for m in metrics:
                            metric_ = m + '/' + os.path.split(exp_path)[-1]
                            if np.any(df['metric'] == metric_):
                                df_ = df[df['metric'] == metric_]
                                df_ = df_.drop(columns='metric')
                                records.append([
                                    t, dset_dir,
                                    os.path.split(exp_path)[-1],
                                    df_.iloc[-1].value
                                ])

    df = pd.DataFrame.from_records(records,
                                   columns=('Types', 'dset', 'Methods', 'F1'))

    return df


if __name__ == "__main__":
    p = configargparse.ArgParser()
    p.add('--types',
          nargs='+',
          default=['Tweezer', 'Cochlea', 'Slitlamp', 'Brain'])
    p.add('--ksp-root-path', default='/home/ubelix/lejeune/runs/ksptrack')
    p.add('--model-root-path', default='/home/ubelix/lejeune/runs/siamese_dec')
    p.add('--save-path', default='table_results.tex')
    p.add('--piovrs', default='1.4')

    cfg = p.parse_args()

    exp_names = {
        'nnpuss': 'pu_piovrs_{}_ph2'.format(cfg.piovrs)
        # 'nnputrue': 'pu_true'
    }

    order = {
        # 'ksp/pu_true': 'KSPTrack/nnPUtrue',
        'ksp/' + exp_names['nnpuss']: 'KSPTrack/nnPUss',
        exp_names['nnpuss']: 'nnPUss',
        'KSP': 'KSPTrack',
        'mic17': 'EEL',
        'gaze2': 'Gaze2Segment',
        'wtp': 'DL-prior',
    }

    path_18 = pjoin(cfg.ksp_root_path, 'plots_results', 'all_self.csv')
    df_18 = pd.read_csv(path_18)
    to_drop = np.arange(2, 14)
    df_18.drop(df_18.columns[to_drop], axis=1, inplace=True)
    df_18 = df_18.set_index(['Types', 'Methods']).rename(
        columns={
            'F1 mean': 'F1',
            'F1 std': '_F1',
            'PR mean': 'PR',
            'PR std': '_PR',
            'RC mean': 'RC',
            'RC std': '_RC'
        })

    df = parse_ksp_exps(cfg.ksp_root_path, cfg.types, exp_names.values())
    df_mean = df.groupby(['Types', 'Methods']).mean()
    df_std = df.groupby(['Types', 'Methods']).std().rename(columns={
        'F1': '_F1',
        'PR': '_PR',
        'RC': '_RC'
    })
    df_ksp = pd.concat((df_mean, df_std), axis=1)

    df = parse_model_exps(cfg.model_root_path, cfg.types, exp_names.values())
    df_mean = df.groupby(['Types', 'Methods']).mean()
    df_std = df.groupby(['Types', 'Methods']).std().rename(columns={
        'F1': '_F1',
    })
    df_model = pd.concat((df_mean, df_std), axis=1)

    # build full table
    df_all = pd.concat((df_ksp, df_model, df_18), axis=0,
                       levels=1).sort_index(0)
    df_all = df_all.round(decimals=2)

    # drop some methods
    df_all = df_all.drop('KSPopt', level='Methods')

    # rename
    df_all = df_all.rename(index=order)

    # add bold field
    df_all['bold'] = False
    for t in cfg.types:
        idx = df_all.loc[t]['F1'].values.argmax()
        df_all.loc[t]['bold'].iloc[idx] = True

    # compute mean over all types
    means = df_all.groupby(['Methods'])['F1'].mean()
    std = df_all.groupby(['Methods'])['_F1'].mean()

    df_all = df_all.groupby(['Types', 'Methods']).apply(myformat)

    # remove dummy columns
    df_all = df_all.drop(columns=['_F1', '_PR', '_RC', 'bold'])

    df_all = df_all.drop(columns=['PR', 'RC'])
    # df_all = pd.concat((df_all, df_sps)).sort_index()
    df_all = df_all.reset_index().pivot('Methods', 'Types')
    df_all = df_all.reindex(order.values())

    # take only F1
    df_all.columns = df_all.columns.droplevel()
    df_all = df_all[cfg.types]

    print(df_all)

    caption = """
    Quantitative results on all datasets. We report the F1 scores and standard deviations.
    """

    print('writing table to {}'.format(cfg.save_path))
    table = df_all.to_latex(
        escape=False,
        column_format='llp{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}',
        multirow=True,
        caption=caption,
        label='tab:results')

    # add horiz line below ours
    with open(cfg.save_path, 'w') as tf:
        for line in table.splitlines():
            if line.startswith('Truth'):
                line += '\n\hline\n'
            if line.startswith('nnPU'):
                line += '\n\hdashline'
            elif line.startswith('\\begin{table}'):
                line = '\\begin{table*}[t]'
            elif line.startswith('\\end{table}'):
                line = '\\end{table*}'
            tf.write(line + '\n')
