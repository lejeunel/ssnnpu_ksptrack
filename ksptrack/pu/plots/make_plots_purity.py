from os.path import join as pjoin
import glob
import pandas as pd
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from tf2pd import tflog2pandas
import seaborn as sns

sns.set_style('darkgrid')

types = ['Tweezer', 'Cochlea', 'Slitlamp', 'Brain']
root_path = pjoin('/home/ubelix/lejeune/runs/siamese_dec')
out_path = '/home/laurent/Documents/papers/aapu/tables/results.tex'

pimul = '1.0'
ur = '0.2'

exp_names = {'aapu': 'aapu*', 'treeaapu': 'treeaapu*'}
metrics = ['aug_purity', 'F1']

order = {
    exp_names['treeaapu']: 'aaPUtrees',
    exp_names['aapu']: 'aaPU',
}


def filter_by_type(dict_, type_, prefix='Dataset'):
    return {k: v for k, v in dict_.items() if k[0][:-1] == prefix + str(type_)}


def filter_by_metric(dict_, metric):
    return {k: v for k, v in dict_.items() if k[-1] == metric}


def filter_by_method(dict_, method):
    # re.split(r"[^a-zA-Z0-9\s]", "foo bar_blub23/x~y'z")
    return {
        k: v
        for k, v in dict_.items()
        if method in re.split(r"[^a-zA-Z0-9\s]", k[1])
    }


def dset_to_type(dset):
    return (types[int(dset[7])])


curves = {}

max_seqs = 4

for i, t in enumerate(types):
    dset_paths = sorted(glob.glob(pjoin(root_path, 'Dataset' + str(i) + '*')))
    for dset_path in dset_paths[:max_seqs]:
        dset_dir = os.path.split(dset_path)[-1]
        exp_dirs = [d[1] for d in os.walk(dset_path)][0]
        for exp_path in exp_names.values():
            exp_path = glob.glob(pjoin(dset_path, exp_path))
            if len(exp_path) > 0:
                exp_path = exp_path[0]
                print(exp_path)
                event_path = glob.glob(pjoin(exp_path, 'event*'))[0]
                if (os.path.exists(event_path)):
                    df = tflog2pandas(event_path)
                    for m in metrics:
                        df_ = df[df['metric'] == m + '/' +
                                 os.path.split(exp_path)[-1]]
                        df_ = df_.drop(columns='metric')
                        curves[(dset_to_type(os.path.split(dset_path)[-1]),
                                os.path.split(dset_path)[-1],
                                re.split(r"[^a-zA-Z0-9\s]",
                                         os.path.split(exp_path)[-1])[0],
                                m)] = df_

df = pd.concat(curves).reset_index()
df = df.drop(columns='level_4')
df.columns = ['Type', 'Dataset', 'Method', 'metric', 'value', 'epoch']

df_ = df[df['metric'] == 'aug_purity']
g = sns.FacetGrid(df_, col='Type', hue='Method', height=5, col_wrap=2)
g.map(sns.lineplot, 'epoch', 'value')
g.set_axis_labels('Epoch', 'Augmented set purity')
g.set_titles(col_template="{col_name}")
g.add_legend()
g.savefig('purity.png')

df_ = df[df['metric'] == 'F1']
g = sns.FacetGrid(df_, col='Type', hue='Method', height=5, col_wrap=2)
g.map(sns.lineplot, 'epoch', 'value')
g.set_axis_labels('Epoch', 'F1')
g.set_titles(col_template="{col_name}")
g.add_legend()
g.savefig('f1.png')
