#!/usr/bin/env python3
from ksptrack.pu.plots.freq_vs_epc import get_all, do_plot
import pandas as pd
import matplotlib.pyplot as plt

import configargparse

if __name__ == "__main__":
    p = configargparse.ArgParser()
    p.add('--root-path', required=True)
    p.add('--train-dir', required=True)
    p.add('--exp-names', nargs='+', required=True)
    p.add('--curves-dir', default='curves_data')
    p.add('--thr', default=0.006)
    p.add('--n-epc', default=5)
    p.add('--rho-pi-err', default=1.0, type=float)
    p.add('--min-epc', default=30)
    p.add('--title', default='')
    p.add('--save', default='')
    cfg = p.parse_args()

    dfs = {exp.split('_')[2]: None for exp in cfg.exp_names}

    for exp_name in cfg.exp_names:

        df_stats, df_tnsr, _ = get_all(cfg.root_path, cfg.train_dir, exp_name,
                                       cfg.curves_dir, cfg.thr, cfg.rho_pi_err,
                                       cfg.min_epc, cfg.n_epc)
        dfs[exp_name.split('_')[2]] = df_stats

    dfs = pd.concat(dfs)
    dfs = dfs.reset_index()
    del dfs['level_1']
    dfs = dfs.rename(columns={'level_0': 'piovrs'})

    fig = do_plot(dfs, cfg.thr, cfg.min_epc, hue='piovrs', legend_hue=True)
    if cfg.title:
        fig.suptitle(cfg.title)

    if cfg.save:
        fig.savefig(cfg.save, dpi=300)
    else:
        plt.show()
