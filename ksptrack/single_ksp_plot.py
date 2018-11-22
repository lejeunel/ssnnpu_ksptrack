import os
from ksptrack import iterative_ksp
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import write_frames_results as writef
from ksptrack.cfgs import cfg

dir_ = ['/home/laurent.lejeune/medical-labeling/Dataset10/results/2017-12-06_14-26-17_exp',
        '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling/Dataset11/results/2017-12-06_16-16-27_exp',
        '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling/Dataset12/results/2017-12-06_17-28-07_exp',
        '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling/Dataset13/results/2017-12-06_20-50-10_exp']

for d in dir_:

    conf_path = os.path.join(d, 'cfg.yml')

    conf = cfg.load_and_convert(conf_path)
    conf.dataOutImageResultDir = 'results'

    # Write result frames
    writef.main(conf)
