from os.path import join as pjoin
import glob
import pandas as pd
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt


# types = ['tweezer', 'cochlea', 'slitlamp', 'brain', 'spleen', 'liver']
types = ['tweezer', 'cochlea', 'slitlamp', 'brain']
root_path = pjoin('/home/ubelix/lejeune/runs/ksptrack')
exp_filter = 'transexp'

records = []

for i, t in enumerate(types):
    dset_paths = sorted(glob.glob(pjoin(root_path, 'Dataset' + str(i) + '*')))
    for dset_path in dset_paths:
        dset_dir = os.path.split(dset_path)[-1]
        exp_paths = sorted(glob.glob(pjoin(dset_path, exp_filter + '*')))
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
                    t, dset_dir, int(100*cfg['ml_down_thr']),
                                  int(100*cfg['ml_up_thr']),
                    df['f1_ksp']
                ])

df = pd.DataFrame.from_records(records,
                               columns=('type', 'dset', 'low', 'high', 'f1'))

# sns.set(style="darkgrid")
sns.catplot(x='low',
            y='f1',
            col='type',
            kind='box',
            data=df,
            hue='high',
            col_wrap=2)
plt.show()
