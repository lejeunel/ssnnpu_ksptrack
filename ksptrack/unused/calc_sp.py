from ksptrack.utils.data_manager import DataManager
from ksptrack.cfgs import params
from os.path import join as pjoin
from ksptrack.utils import my_utils as utls
import os

if __name__ == "__main__":
    p = params.get_params('../cfgs')

    p.add('--root-path', required=True)
    p.add('--sets', nargs='+', required=True)

    cfg = p.parse_args()

    for s in cfg.sets:
        cfg.in_path = pjoin(cfg.root_path, 'data', 'medical-labeling', s)
        cfg.out_path = pjoin(cfg.root_path, 'runs', 'ksptrack', s)

        cfg.precomp_desc_path = pjoin(cfg.in_path, cfg.precomp_dir)
        if (not os.path.exists(cfg.precomp_desc_path)):
            os.makedirs(cfg.precomp_desc_path)

        cfg.frameFileNames = utls.get_images(
            pjoin(cfg.in_path, cfg.frame_dir))
        dm = DataManager(cfg)
        dm.calc_superpix()

