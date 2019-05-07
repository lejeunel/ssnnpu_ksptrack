import os
from ksptrack import iterative_ksp
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import write_frames_results as writef
from cfgs.params import get_params

p = get_params()

p.add('--in-path', required=True)
p.add('--out-path', required=True)

cfg = p.parse_args()

# Run segmentation
conf, logger = iterative_ksp.main(cfg)

# Write result frames
writef.main(conf, logger=logger)
