#!/usr/bin/env python3

from ksptrack.prev_trans_costs import main
from ksptrack.cfgs import params
import numpy as np
from skimage import io

p = params.get_params('../cfgs')
cfg = p.parse_args()

cfg.in_path = '/home/ubelix/artorg/lejeune/data/medical-labeling/Dataset12'
cfg.siam_path = '/home/ubelix/artorg/lejeune/runs/siamese_dec/Dataset12/checkpoints/cp_aapu.pth.tar'
cfg.use_siam_pred = True
cfg.siam_trans = 'lfda'
cfg.aug_method = 'tree'
cfg.n_augs = 100
cfg.cuda = True
cfg.do_scores = False
cfg.do_all = False
cfg.fin = [0, 28, 57, 86, 115]

res = main(cfg)

prev_ims = np.concatenate([
    np.concatenate((s['image'], s['pm'], s['pm_thr'], s['entrance']), axis=1)
    for s in res['images']
],
                          axis=0)
io.imsave('/home/ubelix/artorg/lejeune/runs/test.png', prev_ims)
