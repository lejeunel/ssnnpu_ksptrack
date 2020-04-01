from os.path import join as pjoin
import glob
from pathlib import Path
import yaml
import numpy as np


root_path = '/home/ubelix/lejeune/runs/ksptrack'
exp_filter = 'transexp_up'

for dir_ in Path(root_path).rglob('transexp_up*'):
    print(dir_)
    with open(pjoin(dir_, 'cfg.yml')) as f:
        # cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = yaml.unsafe_load(f)
    cfg['ml_up_thr'] = float(cfg['ml_up_thr'])
    cfg['ml_down_thr'] = float(cfg['ml_down_thr'])
    with open(pjoin(dir_, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg, stream=outfile, default_flow_style=False)



