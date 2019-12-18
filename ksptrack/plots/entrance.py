import os
from os.path import join as pjoin
import pandas as pd
import seaborn as sns
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from ksptrack.plots.entrance_paths import paths

root_dir = '/home/ubelix/runs/ksptrack'
out_dir = '/home/ubelix/runs/ksptrack'
seq_types = ['tweezer', 'cochlea', 'slitlamp', 'brain']
methods_dirs = [
    'autoenc_feats_disk_entr', 'autoenc_feats_ss_entr',
    'autoenc_feats_unet_patch_entr', 'autoenc_feats_darnet_entr',
    'pred_feats_disk_entr', 'pred_feats_ss_entr', 'pred_feats_unet_patch_entr',
    'pred_feats_darnet_entr'
]

seqs = [
    'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04',
    'Dataset05', 'Dataset10', 'Dataset11', 'Dataset12', 'Dataset13',
    'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24',
    'Dataset25', 'Dataset30', 'Dataset31', 'Dataset32', 'Dataset33',
    'Dataset34'
]


def dir_name_to_type(dir_name):
    digit_to_type_mapping = {i: seq_types[i] for i in range(len(seq_types))}
    seqs_dict = {v: [] for _, v in digit_to_type_mapping.items()}

    m = re.search('Dataset(.)', dir_name)
    return seq_types[int(m.group(1))]


def make_dataframe(root_dir, run_dirs, fname, methods):
    # store scores
    pds = []
    for method, dir_ in zip(run_dirs, methods):
        seq_type = dir_name_to_type(os.path.split(root_dir)[-1])
        scores = np.genfromtxt(pjoin(root_dir, dir_, fname),
                                delimiter=',',
                                dtype=None)

        index = pd.MultiIndex.from_tuples([(seq_type, os.path.split(root_dir)[-1])],
                                            names=['seq_type', 'seq_n'])
        columns = [r[0].decode('utf-8') for r in scores]
        scores = [r[1] for r in scores]
        df = pd.DataFrame(scores, index=columns, columns=index)
        df = pd.concat([df], keys=[method], names=['method'], axis=1)
        pds.append(df)
    return pds


dfs = []
for s in seqs:
    df = pd.concat(make_dataframe(pjoin(root_dir, s), methods_dirs,
                                  'scores.csv', methods_dirs),
                   axis=1)
    dfs.append(df)

dfs = pd.concat(dfs, axis=1)

metrics_to_plot = ['f1_ksp', 'tpr_ksp', 'fpr_ksp']

df_to_plot = pd.concat([dfs.loc[m] for m in metrics_to_plot], axis=1)
df_to_plot = df_to_plot.reset_index()

sns.set_style("whitegrid")
fig, ax = plt.subplots(len(metrics_to_plot), 1, figsize=(9, 8))
ax = ax.flatten()

for m, a in zip(metrics_to_plot, ax):
    sns.boxplot(y=m,
                x='seq_type',
                hue='method',
                data=df_to_plot,
                palette="Set3",
                ax=a)
    a.set(ylabel='Score', xlabel='Type', title=m)
    # a.set_yticks(np.linspace(0, 1, 11))
    a.legend(loc='center right', bbox_to_anchor=(1.20, 0.8), borderaxespad=0.)
fig.tight_layout()
# Put the legend out of the figure

path = pjoin(out_dir, 'all.png')
print('saving figure to {}'.format(path))
fig.savefig(path)
