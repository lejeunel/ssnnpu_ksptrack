#!/usr/bin/env python3

from os.path import join as pjoin

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import ImageFont

from ksptrack import prev_trans_costs
from ksptrack.cfgs import params as params_ksp
from ksptrack.pu import params
from ksptrack.pu.plots import freq_vs_epc
import numpy as np
import matplotlib.ticker as plticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend import Legend


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)


if __name__ == "__main__":

    p = params.get_params('..')
    p.add('--root-in-path', required=True)
    p.add('--root-run-path', required=True)
    p.add('--curves-dir', default='curves_data')
    p.add('--dset', type=str, default='33')
    p.add('--fin', type=int, default=46)

    p.add('--epochs', nargs='+', type=int, default=[10, 40, 90])
    p.add('--save-path', default='prevs_conv.png')
    cfg = p.parse_args()

    p_ksp = params_ksp.get_params('../../cfgs')
    p_ksp.add('--model-path', default='')
    p_ksp.add('--in-path', default='')
    p_ksp.add('--do-all', default=False)
    p_ksp.add('--return-dict', default=False)
    p_ksp.add('--fin', nargs='+')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]
    cfg_ksp.use_model_pred = True
    cfg_ksp.aug_df_path = ''
    cfg_ksp.trans = 'lfda'
    cfg_ksp.in_path = pjoin(cfg.root_in_path, 'Dataset' + cfg.dset)
    cfg_ksp.precomp_desc_path = pjoin(cfg_ksp.in_path, 'precomp_desc')
    cfg_ksp.fin = [cfg.fin]
    cfg_ksp.sp_labels_fname = 'sp_labels.npy'
    cfg_ksp.do_scores = True
    cfg_ksp.loc_prior = cfg.loc_prior
    cfg_ksp.coordconv = cfg.coordconv
    cfg_ksp.n_augs = 0
    cfg_ksp.aug_method = cfg.aug_method

    res_pu = {e: {'image': None, 'pm': None} for e in cfg.epochs}
    res_kf = {e: None for e in cfg.epochs}

    cfg.in_path = pjoin(cfg.root_in_path, 'Dataset' + cfg.dset)
    cfg.precomp_dir = 'precomp_desc'
    cfg.fin = [cfg.fin]

    df_init = pd.read_pickle(
        pjoin(cfg.root_run_path, 'Dataset' + cfg.dset, cfg.exp_name,
              cfg.curves_dir, 'ep_0001.p'))
    pi_max = df_init['priors_t'].max()
    sns.set_style('darkgrid')

    im_axes = np.array([None] * len(cfg.epochs))
    for i, e in enumerate(cfg.epochs):

        df = pd.read_pickle(
            pjoin(cfg.root_run_path, 'Dataset' + cfg.dset, cfg.exp_name,
                  cfg.curves_dir, 'ep_{:04d}.p').format(e))

        df = df.drop(columns=['new_xi', 'priors_max', 'priors_t+1'])
        df = df.rename(columns={'priors_t': 'priors'})
        df['epoch'] = e
        res_kf[e] = df

        # cfg_ksp.model_path = pjoin(cfg.root_run_path, 'Dataset' + cfg.dset,
        #                            cfg.exp_name, 'cps',
        #                            'cp_{:04d}.pth.tar'.format(e))
        # cfg_ksp.trans_path = None
        # res_pu[e] = prev_trans_costs.main(cfg_ksp)
        res_pu[e]['image'] = np.ones((512, 512)).astype(int) * 255
        res_pu[e]['pm'] = np.ones((512, 512)).astype(int) * 255

    fig = plt.figure(figsize=(8, 4), dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(ncols=4, nrows=2)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.ones((512, 512)).astype(int) * 255)
    plt.axis('off')

    for i, e in enumerate(res_pu.keys()):
        df_ = res_kf[e].drop(columns=['epoch'])

        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(res_pu[e]['pm'])
        plt.axis('off')
        plt.title('epoch {}/{}'.format(e, df_.shape[0]))

        ax = fig.add_subplot(gs[1, i + 1])
        do_legend = i == 0
        do_ticks = i == 0
        g = sns.lineplot(data=pd.melt(df_, ['frame']),
                         x='frame',
                         y='value',
                         hue='variable',
                         ax=ax,
                         legend=do_legend)
        loc = plticker.MultipleLocator(base=df_.shape[0] / 5)
        g.xaxis.set_major_locator(loc)

        loc = plticker.MultipleLocator(base=(1.1 * pi_max) / 5)
        g.yaxis.set_major_locator(loc)
        g.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        g.xaxis.set_major_formatter(FormatStrFormatter('%i'))
        plt.ylim(0, 1.1 * pi_max)
        g.yaxis.label.set_visible(False)

        if do_legend:
            handles, labels = g.get_legend_handles_labels()
            g.legend_ = None

        if not do_ticks:
            g.set_yticklabels([])

    ax = fig.add_subplot(gs[1, 0])
    ax.add_artist(Legend(ax, handles, labels))
    ax.axis('off')

    print('saving fig to {}'.format(cfg.save_path))
    fig.savefig(cfg.save_path, dpi=400, bbox_inches='tight')
