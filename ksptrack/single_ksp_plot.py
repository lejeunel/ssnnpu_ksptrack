import os
from ksptrack import iterative_ksp
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
#from ksptrack.utils import write_frames_results as writef
from ksptrack.utils import write_frames_results_truth as writef
from ksptrack.cfgs import cfg

conf_path = '/home/laurent.lejeune/medical-labeling/Dataset04/results/2018-08-31_15-52-19_exp/cfg.yml'
conf = cfg.load_and_convert(conf_path)

# Write result frames
writef.main(conf)
