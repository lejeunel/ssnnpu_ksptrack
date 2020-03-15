from ksptrack import prev_trans_costs
from ksptrack.prev_trans_costs import colorize
from ksptrack.cfgs import params
from os.path import join as pjoin
import os
import numpy as np
import pickle as pk
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


if __name__ == "__main__":
    p = params.get_params('../../cfgs')
    p.add('--root-data', required=True)
    p.add('--root-runs', required=True)
    p.add('--fins', nargs='+', type=int, required=True)
    p.add('--dsets', nargs='+', type=str, required=True)
    p.add('--return-dict', default=True)
    p.add('--fin', type=int)
    p.add('--out-data-fname', type=str, default='mic20_prevs.p')
    p.add('--out-img-fname', type=str, default='mic20_prevs.png')
    p.add('--do-all', default=False)
    cfg = p.parse_args()

    out_data_path = pjoin(cfg.root_runs, cfg.out_data_fname)
    out_img_path = pjoin(cfg.root_runs, cfg.out_img_fname)

    if(os.path.exists(out_data_path)):
        print('{} already exists. loading...'.format(out_data_path))
        with open(out_data_path, 'rb') as f:
            all_ = pk.load(f)
    else:

        all_ = []
        for fin, dset in zip(cfg.fins, cfg.dsets):
            cfg.fin = [fin]
            cfg.in_path = pjoin(cfg.root_data, 'Dataset' + str(dset))
            cfg.siam_path = pjoin(cfg.root_runs, 'siamese_dec',
                                'Dataset' + str(dset),
                                'checkpoints',
                                'init_dec.pth.tar')

            ksp = np.load(pjoin(cfg.root_runs, 'ksptrack',
                                'Dataset' + str(dset),
                                'exp_dml',
                                'results.npz'))['ksp_scores_mat']
            segm = colorize(ksp[..., fin].astype(float))

            dict_init = prev_trans_costs.main(cfg)[0]

            cfg.siam_path = pjoin(cfg.root_runs, 'siamese_dec',
                                  'Dataset' + str(dset),
                                  'checkpoints',
                                  'checkpoint_siam.pth.tar')
            dict_last = prev_trans_costs.main(cfg)[0]

            all_.append({'init': dict_init,
                         'last': dict_last,
                         'segm': segm})

        with open(out_data_path, 'wb') as f:
            print('saving prev data to {}'.format(out_data_path))
            pk.dump(all_, f, pk.HIGHEST_PROTOCOL)

    print('generating image to {}'.format(cfg.out_img_fname))
    
    fig = plt.figure(figsize=(24., 24.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(all_), 7),
                     axes_pad=0.05)

    pos = 0
    for all__ in all_:
        im_row = [all__['init']['image'],
                  all__['init']['pm_thr'],
                  all__['init']['clusters'],
                  all__['last']['clusters'],
                  all__['last']['pm_thr'],
                  all__['last']['entrance'],
                  all__['segm']]
        for arr in im_row:
            # Iterating over the grid returns the Axes.
            grid[pos].imshow(arr)
            grid[pos].axis('off')
            pos += 1

    # add text
    titles = ['image', 'initial foreground', 'K-means assignments',
              'DEC assignments',
              'DEC foreground',
              'GMM probabilities',
              'segmentation'
    ]
    for pos in range(7):
        grid[pos].set_title(titles[pos])
    plt.savefig(out_img_path, dpi=100, bbox_inches='tight')


