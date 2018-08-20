from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib
import my_utils as utls
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import learning_dataset as ld
from skimage import (color, segmentation, util, transform, io)
import gazeCsv as gaze
import results_dirs as rd
import shutil as sh
from PIL import Image, ImageFont, ImageDraw

n_decs = 2
dsets_per_type = 1

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

im_path = '/home/laurent.lejeune/medical-labeling/Dataset00/input-frames/frame_0400.png'
unet_out_path = '/home/laurent.lejeune/medical-labeling/Dataset00/precomp_descriptors/gaze_out_example.npz'

unet_out = np.load(unet_out_path)['out']
import pdb; pdb.set_trace()
im = utls.imread(im_path)

#print('Saving to: ' + file_out)
#img.save(file_out)
