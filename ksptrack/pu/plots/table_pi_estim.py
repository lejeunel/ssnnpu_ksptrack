#!/usr/bin/env python3

import numpy as np
from os.path import join as pjoin
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import configargparse
import pandas as pd
from ksptrack.pu.plots.tf2pd import tflog2pandas
from scipy.signal import find_peaks
from ksptrack.utils.base_dataset import BaseDataset
import yaml
import os

types = ['Tweezer', 'Cochlea', 'Slitlamp', 'Brain']
sns.set_style('darkgrid')


def myformat(r):
    if (r['bold'].iat[0]):
        r['MAE'] = '$\\bm{' + r['MAE'].apply(str) + '} \pm ' + r['MAE_'].apply(
            str) + '$'
    else:
        r['MAE'] = '$' + r['MAE'].apply(str) + ' \pm ' + r['MAE_'].apply(
            str) + '$'

    return r


def get_error(root_path, train_dir, exp_name, curve_dir):
    files = sorted(
        glob(pjoin(cfg.root_path, train_dir, exp_name, cfg.curves_dir, '*.p')))

    err_freq, freqs, model_priors, epochs = [], [], [], []

    print('exp path: {}'.format(pjoin(cfg.root_path, train_dir, exp_name)))
    print('loading {} files'.format(len(files)))
    pbar = tqdm(total=len(files))
    for f in files:
        df = pd.read_pickle(f)
        freqs.append(df['model'])
        err_freq.append(np.abs(df['true'] - df['model']).mean())
        model_priors.append(df['model'])
        fname = os.path.splitext(os.path.split(f)[-1])[0]
        ep = int(fname.split('_')[1])
        epochs.append(ep)
        pbar.update(1)
    pbar.close()

    freqs_ = np.array(freqs)
    diff_freqs_ = np.abs(freqs_[1:] - freqs_[:-1]).mean(axis=1)

    df_stats = pd.DataFrame.from_dict({'epoch': epochs, 'err_freq': err_freq})

    # read tensorboard stuff
    df_tnsr = tflog2pandas(
        glob(pjoin(cfg.root_path, train_dir, exp_name, 'event*'))[0])

    key = [
        k for k in df_tnsr['metric'] if k.split('/')[0] == 'var_pseudo_neg'
    ][0]
    df_tnsr = df_tnsr[df_tnsr['metric'] == key]
    df_stats['var_pseudo_neg'] = df_tnsr['value']

    df_stats['below_var_thr'] = df_stats['var_pseudo_neg'] < cfg.thr

    print('getting initial pi')
    with open(pjoin(cfg.root_path, train_dir, exp_name, 'cfg.yml'),
              'r') as infile:
        y = yaml.load(infile, Loader=yaml.FullLoader)
        pi0 = y['init_pi']
        pi_overspec = y['pi_overspec_ratio']

    below_pi0 = [(freq <= pi0).all() for freq in freqs]
    df_stats['below_pi0'] = below_pi0
    df_stats['candidate'] = df_stats['below_var_thr'] & (
        df_stats['epoch'] - 1 > cfg.min_epc) & df_stats['below_pi0']

    converged = np.zeros(df_stats.shape[0]).astype(bool)
    for i in range(df_stats.shape[0]):
        if df_stats.iloc[i:min(i + cfg.n_epc, df_stats.shape[0]
                               )]['candidate'].all():
            converged[min(i + cfg.n_epc, df_stats.shape[0] - 1)] = True
            break

    if converged.sum() < 1:
        converged[-1] = True

    df_stats['converged'] = converged

    idx_min_err = np.argmin(df_stats['err_freq'])
    idx_converged = np.where(df_stats['converged'])[0]
    if len(idx_converged) == 0:
        idx_converged = df.shape[0] - 1
    else:
        idx_converged = idx_converged[0]

    df_stats['convergence'] = ''
    df_stats.loc[idx_min_err, 'convergence'] = 'optima'
    df_stats.loc[idx_converged, 'convergence'] += 'stop'
    df_stats.loc[df_stats.convergence == 'optimastop',
                 'convergence'] = 'optima/stop'

    dfs = df_stats
    ep_converged = df_stats[(
        (df_stats['convergence'] == 'stop')
        | (df_stats['convergence'] == 'optimastop'))]['epoch'].item() - 1

    ep_start = dfs.iloc[0]['epoch']
    ep = dfs[(dfs['epoch'] == ep_converged)]['epoch'] - ep_start
    err = err_freq[ep.item()]

    return err, pi0, pi_overspec


def main(cfg):

    # max_seqs = 4
    # records = []
    # for i, t in enumerate(types):
    #     dset_paths = sorted(
    #         glob(pjoin(cfg.root_path, 'Dataset' + str(i) + '*')))
    #     for dset_path in dset_paths[:max_seqs]:
    #         exp_names = glob(
    #             pjoin(dset_path, cfg.exp_name_prefix + '*' + 'ph1'))
    #         for exp_name in exp_names:
    #             e, pi0, piovrs = get_error(cfg.root_path, dset_path, exp_name,
    #                                        cfg.curves_dir)
    #             records.append([t, os.path.split(dset_path)[-1], e, piovrs])

    # df = pd.DataFrame.from_records(records,
    #                                columns=('Types', 'dset', 'MAE', 'piovrs'))
    # df.to_pickle('pi_estim.p')

    df = pd.read_pickle('pi_estim.p')
    df_mean = df.groupby(['Types', 'piovrs']).mean()
    df_std = df.groupby(['Types',
                         'piovrs']).std().rename(columns={'MAE': 'MAE_'})
    df = pd.concat((df_mean, df_std), axis=1)
    df = df.round(decimals=2)

    # add bold field
    df['bold'] = False
    for t in types:
        idx = df.loc[t]['MAE'].values.argmin()
        df.loc[t]['bold'].iloc[idx] = True

    df = df.groupby(['Types', 'piovrs']).apply(myformat)

    df = df.drop(columns=['MAE_', 'bold'])
    df = df.reset_index().pivot('Types', 'piovrs')
    df.columns = df.columns.droplevel()
    print(df)

    caption = """
    Mean Absolute Error (MAE) of estimated class-priors for all types for different initial conditions.
    We show the means and standard deviations.
    Lowest are in bold.
    """

    print('writing table to {}'.format(cfg.out_path))
    table = df.to_latex(escape=False,
                        column_format='l*{3}',
                        multirow=True,
                        caption=caption,
                        label='tab:mae')

    with open(cfg.out_path, 'w') as tf:
        for line in table.splitlines():
            tf.write(line + '\n')
    return df


if __name__ == "__main__":
    p = configargparse.ArgParser()

    p.add('--root-path', required=True)
    p.add('--out-path', required=True)
    p.add('--exp-name-prefix', default='pu_piovrs')
    p.add('--curves-dir', default='curves_data')

    p.add('--thr', default=0.003)
    p.add('--n-epc', default=10)
    p.add('--min-epc', default=40)
    p.add('--title', default='')
    p.add('--save', default='')
    cfg = p.parse_args()

    main(cfg)
