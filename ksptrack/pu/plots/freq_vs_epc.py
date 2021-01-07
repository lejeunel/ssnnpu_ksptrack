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


def do_plot(df, thr, min_epc, hue='dset', title='', legend_hue=False):
    sns.set_theme()
    scrtmarkers = {'optima': 'o', 'stop': 'X', 'optima/stop': 'D'}
    plt.subplot(211)
    if title:
        plt.title(title)

    gsc = sns.scatterplot(data=df[df['convergence'] != ''],
                          x='epoch',
                          y='err_prior',
                          hue=hue,
                          markers=scrtmarkers,
                          s=100,
                          style='convergence',
                          legend=True)
    # get current handles and labels
    current_handles, current_labels = plt.gca().get_legend_handles_labels()

    # remove title
    t_ = [(h, l) for h, l in zip(current_handles, current_labels)
          if (('optima' in l) or ('stop' in l))]
    current_handles = [t[0] for t in t_]
    current_labels = [t[1] for t in t_]
    conv_leg = plt.legend(current_handles,
                          current_labels,
                          bbox_to_anchor=(1.0, 1.0),
                          loc='upper right')

    g = sns.lineplot(data=df,
                     x='epoch',
                     y='err_prior',
                     hue=hue,
                     legend=legend_hue)
    if legend_hue:

        # get handles of "hue"
        n_hue = df[hue].unique().size
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        current_handles = current_handles[:n_hue]
        current_labels = current_labels[:n_hue]
        plt.legend(current_handles,
                   current_labels,
                   bbox_to_anchor=(0.75, 1.),
                   loc='upper right')
        # We need this because the 2nd call to legend() erases the first #
        g.add_artist(conv_leg)

    g.set(xlabel=None)
    plt.ylabel("Mean absolute error \n of priors")
    plt.subplot(212)
    g = sns.lineplot(data=df,
                     x='epoch',
                     y='var_pseudo_neg',
                     hue=hue,
                     legend=False)
    plt.axhline(y=thr, color='r', linestyle='--')
    plt.axvline(x=min_epc, color='g', linestyle='--')
    g = sns.scatterplot(data=df[df['convergence'] != ''],
                        x='epoch',
                        y='var_pseudo_neg',
                        hue=hue,
                        legend=False,
                        markers=scrtmarkers,
                        s=100,
                        style='convergence')
    plt.ylabel("variance of\n pseudo negatives")

    return plt.gcf()


def get_all(root_path, train_dir, exp_name, curves_dir, thr, rho_pi_err_thr,
            min_epc, n_epc):
    freqs = []
    err_prior = []
    err_freq = []
    mean_freq = []
    max_freq = []
    priors = []
    prior_model_diff = []
    pi_0 = []
    model_priors = []
    epochs = []

    files = sorted(
        glob(pjoin(root_path, train_dir, exp_name, curves_dir, '*.p')))

    print('exp path: {}'.format(pjoin(root_path, train_dir, exp_name)))
    print('loading {} files'.format(len(files)))
    pbar = tqdm(total=len(files))
    for f in files:
        df = pd.read_pickle(f)
        freqs.append(df['clf'])
        err_freq.append(np.abs(df['true'] - df['clf']).mean())
        mean_freq.append(np.mean(df['clf']))
        max_freq.append(np.max(df['clf']))
        p = df['priors_t']
        err_prior.append(np.abs(df['true'] - p).mean())
        priors.append(p)
        prior_model_diff.append(np.abs((df['clf'] - p).mean()))
        pi_0.append(np.mean(p))
        model_priors.append(df['clf'])
        fname = os.path.splitext(os.path.split(f)[-1])[0]
        ep = int(fname.split('_')[1])
        epochs.append(ep)
        pbar.update(1)
    pbar.close()

    freqs_ = np.array(freqs)
    diff_freqs_ = np.abs(freqs_[1:] - freqs_[:-1]).mean(axis=1)
    mean_freqs_ = np.mean(freqs_, axis=1)
    max_freqs_ = np.max(freqs_, axis=1)
    rel_diff_freqs_ = diff_freqs_ / mean_freqs_[1:]

    df_stats = pd.DataFrame.from_dict({
        'epoch':
        epochs,
        'err_prior':
        err_prior,
        'err_freq':
        err_freq,
        'mean_freq':
        mean_freqs_,
        'max_freq':
        max_freqs_,
        'prior_model_diff':
        np.array(prior_model_diff),
        'rel_diff_freq':
        np.concatenate(([np.nan], rel_diff_freqs_))
    })

    # df_stats['epoch'] = df_stats['epoch'] - df_stats['epoch'].min()

    # read tensorboard stuff
    df_tnsr = tflog2pandas(
        glob(pjoin(root_path, train_dir, exp_name, 'event*'))[0])

    key = [
        k for k in df_tnsr['metric'] if k.split('/')[0] == 'var_pseudo_neg'
    ][0]
    df_tnsr = df_tnsr[df_tnsr['metric'] == key]
    df_stats['var_pseudo_neg'] = df_tnsr['value']

    df_stats['below_var_thr'] = df_stats['var_pseudo_neg'] < thr

    print('getting initial pi')
    with open(pjoin(root_path, train_dir, exp_name, 'cfg.yml'), 'r') as infile:
        pi0 = yaml.load(infile, Loader=yaml.FullLoader)['init_pi']

    below_pi0 = [(freq <= pi0).all() for freq in freqs]
    rho_pi_err = np.array([(np.abs(rho - pi) / pi0).mean()
                           for rho, pi in zip(freqs, priors)])

    df_stats['rho_pi_err'] = rho_pi_err < rho_pi_err_thr
    df_stats['below_pi0'] = below_pi0
    # df_stats['candidate'] = df_stats['below_var_thr'] & (
    #     df_stats['epoch'] - 1 > min_epc) & df_stats['rho_pi_err']
    df_stats['candidate'] = df_stats['below_var_thr'] & (
        df_stats['epoch'] >
        min_epc) & df_stats['rho_pi_err'] & df_stats['below_pi0']

    converged = np.zeros(df_stats.epoch.max()).astype(bool)
    # find points where all frames are below pi_0
    #
    # find points in below_thr binary signal
    for e in range(min_epc, df_stats.epoch.max()):
        range_ = df_stats.epoch >= e
        range_ &= df_stats.epoch < min(e + n_epc, df_stats.epoch.max())
        if df_stats[range_]['candidate'].all():
            converged[min(e - 1, df_stats.epoch.max() - 1)] = True
            break

    if converged.sum() < 1:
        converged[-1] = True

    df_stats['converged'] = converged

    idx_min_err = np.argmin(df_stats['err_prior'])
    idx_converged = np.where(df_stats['converged'])[0]
    if len(idx_converged) == 0:
        idx_converged = df_stats.shape[0]
    else:
        idx_converged = idx_converged[0]

    df_stats['convergence'] = ''
    df_stats.loc[idx_min_err, 'convergence'] = 'optima'
    df_stats.loc[idx_converged, 'convergence'] += 'stop'
    df_stats.loc[df_stats.convergence == 'optimastop',
                 'convergence'] = 'optima/stop'

    return df_stats, df_tnsr, priors


def main(cfg):

    priors = {d: [] for d in cfg.train_dirs}

    dfs = {}

    for d in cfg.train_dirs:

        df_stats, df_tnsr, priors_ = get_all(cfg.root_path, d, cfg.exp_name,
                                             cfg.curves_dir, cfg.thr,
                                             cfg.rho_pi_err, cfg.min_epc,
                                             cfg.n_epc)
        priors[d] = priors_

        dfs[d] = df_stats

    dfs = pd.concat(dfs)
    dfs = dfs.reset_index()
    del dfs['level_1']
    dfs = dfs.rename(columns={'level_0': 'dset'})

    fig = do_plot(dfs, cfg.thr, cfg.min_epc, hue='dset')

    if cfg.title:
        fig.suptitle(cfg.title, y=0.92)
    if cfg.save:
        # plt.tight_layout()
        fig.set_size_inches(8.5, 5.5)
        fig.savefig(cfg.save, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    ep_converged = [
        dfs[(dfs['dset'] == d)
            & ((dfs['convergence'] == 'stop')
               | (dfs['convergence'] == 'optima/stop'))]['epoch'].item()
        for d in cfg.train_dirs
    ]
    ep_converged_ = []
    priors_ = []
    for d, e in zip(cfg.train_dirs, ep_converged):
        ep_start = dfs[(dfs['dset'] == d)].iloc[0]['epoch']
        ep = dfs[(dfs['dset'] == d) & (dfs['epoch'] == e)]['epoch'] - ep_start
        priors_.append(priors[d][ep.item()])
        ep_converged_.append(ep + 1)

    print(ep_converged)

    return priors_, ep_converged


if __name__ == "__main__":
    p = configargparse.ArgParser()

    p.add('--root-path', required=True)
    p.add('--train-dirs', nargs='+', required=True)
    p.add('--exp-name', required=True)
    p.add('--curves-dir', default='curves_data')
    p.add('--thr', default=0.007)
    p.add('--n-epc', default=10)
    p.add('--rho-pi-err', default=999, type=float)
    p.add('--min-epc', default=30)
    p.add('--title', default='')
    p.add('--save', default='')
    cfg = p.parse_args()

    main(cfg)
