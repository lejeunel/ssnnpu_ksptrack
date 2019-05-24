import os
from os.path import join as pjoin
from ksptrack import iterative_ksp
from ksptrack.utils.data_manager import DataManager
import ksptrack.sp_manager as spm
from ksptrack.utils import my_utils as utls
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import write_frames_results as writef
from cfgs.params import get_params

p = get_params()

p.add('--in-path')
p.add('--out-path')

cfg = p.parse_args()

cfg.in_path = '/home/laurent.lejeune/medical-labeling/Dataset00'
cfg.out_path = '/home/laurent.lejeune/medical-labeling/Dataset00'

cfg.bag_jobs = 4
cfg.bag_t = 100

# Make frame file names
cfg.frameFileNames = utls.get_images(
    os.path.join(cfg.in_path, cfg.frame_dir))

locs2d = utls.readCsv(
    os.path.join(cfg.in_path, cfg.locs_dir, cfg.csv_fname))

# cfg.precomp_desc_path = cfg.out_path
cfg.precomp_desc_path = pjoin(cfg.in_path, 'precomp_desc')
if (not os.path.exists(cfg.precomp_desc_path)):
    os.makedirs(cfg.precomp_desc_path)

# ---------- Descriptors/superpixel costs
dm = DataManager(cfg)
dm.calc_superpix()

locs2d_sps = utls.locs2d_to_sps(locs2d, dm.labels)

# dm.calc_sp_feats_unet_gaze_rec(locs2d, save_dir=cfg.precomp_desc_path)

sps_man = spm.SuperpixelManager(
    dm, cfg, with_flow=cfg.use_hoof, init_mode=cfg.sp_trans_init_mode,
    init_radius=cfg.sp_trans_init_radius)

dm.calc_pm(
    np.array(locs2d_sps), all_feats_df=dm.sp_desc_df, mode='foreground')

frames=[10, 30, 60, 90]
pm = dm.get_pm_array(frames=frames)
plt.imshow(pm[..., 60] > cfg.pm_thr);plt.show()

