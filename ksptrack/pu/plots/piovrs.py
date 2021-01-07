#!/usr/bin/env python3
from ksptrack.pu.plots.table_results import parse_ksp_exps
import configargparse
import seaborn as sns
import matplotlib.pyplot as plt


def method_to_param(df, field='Methods'):
    params = [
        float(r[field].split('/')[-1].split('_')[-2])
        for _, r in df.iterrows()
    ]

    df[field] = params
    df = df.rename(columns={field: 'param'})
    return df


def parse_all(cfg):
    df = parse_ksp_exps(cfg.root_path, [cfg.type], cfg.exp_names)
    df = method_to_param(df)

    return df


def main(cfg):

    sns.set_theme()
    df = parse_all(cfg)
    fig = sns.lineplot(
        data=df,
        x="param",
        y="F1",
        # hue="event",
        err_style="bars",
        ci=68)

    plt.xlabel('initial prior overspecification')

    if cfg.title:
        plt.title(cfg.title)

    if cfg.save:
        plt.tight_layout()
        plt.savefig(cfg.save, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    p = configargparse.ArgParser()

    p.add('--root-path', required=True)
    p.add('--train-dirs', nargs='+', required=True)
    p.add('--exp-names', nargs='+', required=True)
    p.add('--type', default='')
    p.add('--title', default='')
    p.add('--save', default='')
    cfg = p.parse_args()

    main(cfg)
