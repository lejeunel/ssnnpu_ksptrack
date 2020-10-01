from os.path import join as pjoin
import glob
import pandas as pd
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from ksptrack.exps import results_dirs as rd
import numpy as np

types = ['Tweezer', 'Cochlea', 'Slitlamp', 'Brain']
root_path = pjoin('/home/ubelix/lejeune/runs/ksptrack')
out_path = '/home/laurent/Documents/papers/aapu/tables/results.tex'

pimul = '1.0'
ur = '0.2'

exp_names = {
    'bce': 'bce_pimul_{}_pr_bag'.format(pimul),
    'nnpu': 'pu_pimul_{}_pr_bag'.format(pimul),
    'aapu': 'aapu_pimul_{}_ur_{}_pr_bag'.format(pimul, ur),
    'treeaapu': 'treeaapu_pimul_{}_ur_{}_pr_bag'.format(pimul, ur)
}

order = {
    exp_names['treeaapu']: 'KSPTrack/aaPUtrees',
    exp_names['aapu']: 'KSPTrack/aaPU',
    exp_names['nnpu']: 'KSPTrack/nnPU',
    exp_names['bce']: 'KSPTrack/BBCE',
    'KSP': 'KSPTrack',
    'mic17': 'EEL',
    'gaze2': 'Gaze2Segment',
    'wtp': 'DL-prior',
}

path_18 = pjoin(root_path, 'plots_results', 'all_self.csv')
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

records = []

max_seqs = 4

for i, t in enumerate(types):
    dset_paths = sorted(glob.glob(pjoin(root_path, 'Dataset' + str(i) + '*')))
    for dset_path in dset_paths[:max_seqs]:
        dset_dir = os.path.split(dset_path)[-1]
        exp_paths = [pjoin(dset_path, x) for x in exp_names.values()]
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
                    t, dset_dir,
                    os.path.split(exp_path)[-1], df['f1_ksp'], df['pr_ksp'],
                    df['rc_ksp']
                ])

df = pd.DataFrame.from_records(records,
                               columns=('Types', 'dset', 'Methods', 'F1', 'PR',
                                        'RC'))
df_mean = df.groupby(['Types', 'Methods']).mean()
df_std = df.groupby(['Types', 'Methods']).std().rename(columns={
    'F1': '_F1',
    'PR': '_PR',
    'RC': '_RC'
})

# build full table
df = pd.concat((df_mean, df_std), axis=1)
df_all = pd.concat((df, df_18), axis=0, levels=1).sort_index(0)
df_all = df_all.round(decimals=2)

# drop some methods
df_all = df_all.drop('KSPopt', level='Methods')

# rename
df_all = df_all.rename(index=order)

# add bold field
df_all['bold'] = False
for t in types:
    idx = df_all.loc[t]['F1'].values.argmax()
    df_all.loc[t]['bold'].iloc[idx] = True


# add bold tags
def myformat(r):
    if (r['bold'].iat[0]):
        r['F1'] = '$\\bm{' + r['F1'].apply(str) + '} \pm ' + r['_F1'].apply(
            str) + '$'
    else:
        r['F1'] = '$' + r['F1'].apply(str) + ' \pm ' + r['_F1'].apply(
            str) + '$'

    r['PR'] = '$' + r['PR'].apply(str) + ' \pm ' + r['_PR'].apply(str) + '$'
    r['RC'] = '$' + r['RC'].apply(str) + ' \pm ' + r['_RC'].apply(str) + '$'

    return r


# compute mean over all types
means = df_all.groupby(['Methods'])['F1'].mean()
std = df_all.groupby(['Methods'])['_F1'].mean()

df_all = df_all.groupby(['Types', 'Methods']).apply(myformat)

# remove dummy columns
df_all = df_all.drop(columns=['_F1', '_PR', '_RC', 'bold'])

df_all = df_all.reset_index().pivot('Methods', 'Types')
df_all = df_all.reindex([v for v in order.values()])

# take only F1
# df_all = df_all[['F1', 'mean', 'std']]
df_all = df_all[['F1']]

df_all.columns = df_all.columns.droplevel()
print(df_all)

caption = """
Quantitative results on all datasets. We report the F1 scores and standard deviations.
"""

print('writing table to {}'.format(out_path))
table = df_all.to_latex(
    escape=False,
    column_format='llp{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}',
    multirow=True,
    caption=caption,
    label='tab:results')

# add horiz line below ours
with open(out_path, 'w') as tf:
    for line in table.splitlines():
        if line.startswith('KSPTrack/nnPU'):
            line += '\n\hdashline'
        tf.write(line + '\n')
