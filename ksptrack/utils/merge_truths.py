import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from skimage import io

root_dir = '/home/laurent.lejeune/medical-labeling/Dataset04/original'
left_dir = 'ground_truth-frames-left'
right_dir = 'ground_truth-frames-right'
both_dir = 'ground_truth-frames'
out_dir = os.path.join(root_dir, both_dir)
mask_files = 'frame_*.png'

right_frames = sorted(glob.glob(os.path.join(root_dir, right_dir, mask_files)))
left_frames = sorted(glob.glob(os.path.join(root_dir, left_dir, mask_files)))

if(not os.path.exists(out_dir)):
    os.makedirs(out_dir)

for f_left, f_right in zip(left_frames, right_frames):
    f_left_ = io.imread(f_left)
    f_right_ = io.imread(f_right)
    f_both_ = f_left_ + f_right_
    f_both = os.path.join(out_dir, os.path.split(f_left)[1])
    io.imsave(f_both, f_both_)
    print('frame: {}'.format(f_both))


