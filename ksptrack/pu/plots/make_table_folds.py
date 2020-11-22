from os.path import join as pjoin
import glob
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from ksptrack.exps import results_dirs as rd
import numpy as np

types = ['Tweezer', 'Cochlea', 'Slitlamp', 'Brain']
root_path = pjoin('/home/ubelix/lejeune/runs/dec_folds')
out_path = '/home/laurent/Documents/papers/lejeune_miccai20/table_folds.tex'
exp_filter = 'fold_'

records = []
for i, t in enumerate(types):
    dset_paths = sorted(glob.glob(pjoin(root_path, t + '*')))
    for dset_path in dset_paths:
        exp_paths = sorted(glob.glob(pjoin(dset_path, '*')))
        for exp_path in exp_paths:
            folds_paths = sorted(glob.glob(pjoin(exp_path, '*')))
            for f in folds_paths:
                score_path = pjoin(f, 'scores.csv')
                if (os.path.exists(score_path)):
                    df = pd.read_csv(score_path,
                                     index_col=0,
                                     header=None,
                                     squeeze=True)

                    records.append([
                        t,
                        os.path.split(exp_path)[-1],
                        os.path.split(f)[-1], df['f1']
                    ])

df = pd.DataFrame.from_records(records,
                               columns=('Types', 'Methods', 'Fold', 'F1'))
df_mean = df.groupby(['Types', 'Methods']).mean()
df_std = df.groupby(['Types', 'Methods']).std().rename(columns={'F1': '_F1'})

# build full table
df = pd.concat((df_mean, df_std), axis=1)
df = df.round(decimals=2)

# rename
df = df.rename(index={'dml': 'KSPTrack/DEC'})
df = df.rename(index={'lfda': 'KSPTrack/GMM'})

# add bold field
df['bold'] = False
for t in df.index.unique(level='Types'):
    idx = df.loc[t]['F1'].values.argmax()
    df.loc[t]['bold'].iloc[idx] = True


# add bold tags
def myformat(r):
    if (r['bold'].iat[0]):
        r['F1'] = '$\\bm{' + r['F1'].apply(str) + '} \pm ' + r['_F1'].apply(
            str) + '$'
    else:
        r['F1'] = '$' + r['F1'].apply(str) + ' \pm ' + r['_F1'].apply(
            str) + '$'

    # r['PR'] = '$' + r['PR'].apply(str) + ' \pm ' + r['_PR'].apply(str) + '$'
    # r['RC'] = '$' + r['RC'].apply(str) + ' \pm ' + r['_RC'].apply(str) + '$'

    return r


df = df.groupby(['Types', 'Methods']).apply(myformat)

# remove dummy columns
df = df.drop(columns=['_F1', 'bold'])

df = df.reset_index().pivot('Methods', 'Types')

# take only F1
df = df[['F1']]

# sort by methods
order = ['KSPTrack/DEC', 'KSPTrack/GMM']
df = df.reindex(order)

df.columns = df.columns.droplevel()

caption = """
Quantitative results on all datasets. We report the F1 scores and standard deviations.
"""

print('writing table to {}'.format(out_path))
# with open(out_path, 'w') as tf:
# tf.write(df_all.to_latex(escape=False,
table = df.to_latex(escape=False,
                    column_format='llp{1.8cm}',
                    multirow=True,
                    caption=caption,
                    label='tab:results')

# add horiz line below ours
with open(out_path, 'w') as tf:
    for line in table.splitlines():
        if line.startswith('KSPTrack/GMM'):
            line += '\n\hdashline'
        tf.write(line + '\n')
