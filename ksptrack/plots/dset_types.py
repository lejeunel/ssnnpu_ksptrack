from ksptrack.cfgs import params
from os.path import join as pjoin
from ksptrack.utils.base_dataset import BaseDataset
from skimage import io
import os
from os.path import join as pjoin
import tqdm
import pandas as pd


# add bold tags
def myformat(r):
    color = r['Color'].iloc[0]
    if color:
        r['Color'] = '\\cmark'
    else:
        r['Color'] = '\\xmark'

    return r


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--dsets-idx', nargs='+', required=True)

    cfg = p.parse_args()

    dset_dirs = ['Dataset' + idx for idx in cfg.dsets_idx]

    types = {'0': 'Tweezer', '1': 'Cochlea', '2': 'Slitlamp', '3': 'Brain'}
    colors = {'0': True, '1': False, '2': True, '3': False}
    idx = ['A', 'B', 'C', 'D']

    loaders = [BaseDataset(pjoin(cfg.in_root, dir_)) for dir_ in dset_dirs]

    fields = []

    for l in loaders:
        prefix = os.path.split(l.root_path)[-1][-2]
        f = {
            # 'Type': types[prefix],
            # 'Name': idx[int(os.path.split(l.root_path)[-1][-2])],
            'Height': l[0]['image'].shape[0],
            'Width': l[0]['image'].shape[1],
            'Color': colors[prefix]
        }

        fields.append(f)

    midx = pd.MultiIndex.from_product([[t for t in types.values()], idx],
                                      names=['Type', 'Name'])
    df = pd.DataFrame(fields, index=midx, columns=['Height', 'Width', 'Color'])
    # df = df.reset_index()
    df = df.groupby(['Type', 'Name']).apply(myformat)
    df.index.names = [None, None]

    print(df)

    out_path = pjoin(cfg.out_root, 'dset_types.tex')
    caption = """
    Overview of datasets.
    """

    print('writing table to {}'.format(out_path))
    table = df.to_latex(
        escape=False,
        column_format='llp{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}p{1.8cm}',
        multirow=True,
        multicolumn=True,
        # bold_rows=True,
        # index=False,
        # sparsify=False,
        caption=caption,
        label='tab:dataset_stats')

    with open(out_path, 'w') as tf:
        tf.write(table)
