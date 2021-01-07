import glob
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from ksptrack import prev_trans_costs
from ksptrack.cfgs import params
from skimage import color, io
from ksptrack.pu.im_utils import colorize
from PIL import Image, ImageDraw, ImageFont
import pickle
import os

if __name__ == "__main__":
    p = params.get_params('../../cfgs')
    p.add('--root-in-path', required=True)
    p.add('--root-run-path', required=True)
    p.add('--pu-run-dir', default='siamese_dec')
    p.add('--puksp-run-dir', default='ksptrack')
    p.add('--dsets', nargs='+', type=str, default=['02', '10', '23', '31'])
    p.add('--fin', nargs='+', type=int, default=[0, 49, 57, 46])
    p.add('--save-path', default='prevs.png')
    p.add('--save-path-pk', default='prevs.p')
    cfg = p.parse_args()

    cfg.csv_fname = 'video1.csv'
    cfg.locs_dir = 'gaze-measurements'
    cfg.coordconv = False
    cfg.trans = 'lfda'
    cfg.aug_method = 'none'
    cfg.do_scores = False
    cfg.trans_path = None
    cfg.use_model_pred = True
    cfg.n_augs = 0
    cfg.aug_df_path = ''
    cfg.do_all = False

    if os.path.exists(cfg.save_path_pk):
        print('loading ', cfg.save_path_pk)
        r = pickle.load(open(cfg.save_path_pk, "rb"))
        res_pu = r['pu']
        res_ksp = r['ksp']
    else:

        res_pu = {d: None for d in cfg.dsets}
        res_ksp = {d: None for d in cfg.dsets}

        for fin, dset in zip(cfg.fin, cfg.dsets):

            exp_name = cfg.exp_name
            in_path = pjoin(cfg.root_in_path, 'Dataset' + dset)

            ksp_im = colorize(
                io.imread(
                    pjoin(cfg.root_run_path, cfg.puksp_run_dir,
                          'Dataset' + dset, exp_name, 'results',
                          'im_{:04d}.png'.format(fin))))
            res_ksp[dset] = ksp_im
            cps = sorted(
                glob.glob(
                    pjoin(cfg.root_run_path, cfg.pu_run_dir, 'Dataset' + dset,
                          exp_name, 'cps', '*.pth.tar')))

            cfg.in_path = in_path
            cfg.model_path = cps[-1]
            cfg.fin = [fin]

            res_pu[dset] = prev_trans_costs.main(cfg)

        print('saving to ', cfg.save_path_pk)
        pickle.dump({
            'pu': res_pu,
            'ksp': res_ksp
        }, open(cfg.save_path_pk, "wb"))

    n_cols = 4
    fig = plt.figure()
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(len(cfg.dsets), n_cols),
                     axes_pad=0.02)

    # get a font
    fnt = ImageFont.truetype("DejaVuSans.ttf", 60)

    strs = ['', 'nnPUss', 'nnPUss/thr', 'KSPTrack/nnPUss']
    pos = 0
    for i, d in enumerate(cfg.dsets):
        for j, arr in enumerate([
                res_pu[d]['images'][0]['image'], res_pu[d]['images'][0]['pm'],
                res_pu[d]['images'][0]['pm_thr'], res_ksp[d]
        ]):
            grid[pos].imshow(arr)
            grid[pos].axis('off')
            if i == 0:
                grid[pos].set_title(strs[j], fontsize=8)
            pos += 1

    print('saving fig to {}'.format(cfg.save_path))
    plt.savefig(cfg.save_path, dpi=400, bbox_inches='tight')
