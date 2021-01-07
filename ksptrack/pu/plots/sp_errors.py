from ksptrack.cfgs import params
from os.path import join as pjoin
from ksptrack.utils.base_dataset import BaseDataset
from skimage.measure import regionprops
import os
import numpy as np
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import tqdm
import pandas as pd


def myformat_tex(r):
    r['F1'] = '$' + r['F1'].apply(str) + ' \pm ' + r['_F1'].apply(str) + '$'

    return r


def myformat(r):
    r['F1'] = r['F1'].apply(str) + ' ± ' + r['_F1'].apply(str)

    return r


def main(cfg):
    for idx in cfg.dset_idx:
        run_path = pjoin(cfg.out_root, 'Dataset' + idx)
        if (not os.path.exists(run_path)):
            os.makedirs(run_path)

        df_path = pjoin(run_path, 'scores.csv')

        print('run_path: {}'.format(run_path))

        if (not os.path.exists(df_path)):
            truths = []
            truths_sp = []
            in_path = pjoin(cfg.in_root, 'Dataset' + idx)
            dl = BaseDataset(in_path)
            pbar = tqdm.tqdm(total=len(dl))
            for s in dl:
                labels = s['labels']
                truth = s['label/segmentation']
                regions = regionprops(labels + 1, intensity_image=truth)
                pos = np.array([p['mean_intensity'] > 0.5
                                for p in regions])[..., None]
                mapping = np.concatenate((np.unique(labels)[..., None], pos),
                                         axis=1)

                _, ind = np.unique(labels, return_inverse=True)
                truth_sp = mapping[ind, 1:].reshape(labels.shape)
                truths.append(truth)
                truths_sp.append(truth_sp)
                pbar.update(1)
            pbar.close()

            print('computing f1 to {}'.format(df_path))

            f1 = f1_score(
                np.array(truths).ravel(),
                np.array(truths_sp).ravel())
            data = {'f1': f1}
            df = pd.Series(data)
            df.to_csv(df_path)
        else:
            print('score file {} exists'.format(df_path))


def get_df(cfg):
    records = []
    for idx in cfg.dset_idx:
        run_path = pjoin(cfg.out_root, 'Dataset' + idx)
        df_path = pjoin(run_path, 'scores.csv')

        df = pd.read_csv(df_path, index_col=0, header=None, squeeze=True)
        records.append((params.datasetdir_to_type('Dataset' + idx),
                        'Dataset' + idx, df['f1']))

    df_all = pd.DataFrame.from_records(records,
                                       columns=('Types', 'dset', 'F1'))
    df_mean = df_all.groupby(['Types']).mean()
    df_std = df_all.groupby(['Types']).std().rename(columns={'F1': '_F1'})

    # build full table
    df_all = pd.concat((df_mean, df_std), axis=1)
    df_all = df_all.round(decimals=2)

    return df_all


def make_tables(cfg):
    df_all = get_df(cfg)
    df_all_tex = df_all.groupby(['Types'
                                 ]).apply(myformat_tex).drop(columns=['_F1'])

    df_all_tex.index.names = [None]

    latex_path = cfg.out_tex

    latex = df_all_tex.to_latex(
        escape=False,
        column_format='lp{1.8cm}',
        multirow=True,
        caption=
        'For each type of sequence, we report the maximum F1-score achievable given the early stage superpixel segmentation.',
        label='tab:sp_errors')

    print('writing to {}'.format(latex_path))

    with open(latex_path, 'w') as tf:
        tf.write(latex)

    df_all_html = df_all.apply(myformat).drop(columns=['_F1'])
    html_path = pjoin(cfg.out_root, 'sp_errors.html')
    html = df_all_html.to_html()
    with open(html_path, 'w') as tf:
        tf.write(html)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--out-tex', required=True)
    p.add('--in-root', required=True)
    p.add('--dset-idx', nargs='+', required=True)

    cfg = p.parse_args()

    main(cfg)
    make_tables(cfg)
