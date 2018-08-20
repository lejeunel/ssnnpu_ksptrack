import glob
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
import numpy as np
import matplotlib.pyplot as plt
import my_utils as utls
import shutil
from skimage.transform import resize
from skimage.io import imsave

dir_root = '/home/laurent.lejeune/medical-labeling/'

dir_dataset = 'Dataset10'
ims = sorted(glob.glob(os.path.join(dir_root,dir_dataset,'input-frames','*.png')))
gts = sorted(glob.glob(os.path.join(dir_root,dir_dataset,'ground_truth-frames','*.png')))

#size/ratio
out_size = [546,720]
out_ratio = out_size[1]/out_size[0]
top_left = [0,0]
bot_right = [512,680]
ratio = (top_left[1]-bot_right[1])/(top_left[0]-bot_right[0])

#f_ind = 50
#im = utls.imread(ims[f_ind])
#gt = utls.imread(gts[f_ind])
#plt.subplot(121)
#plt.imshow(im)
#plt.subplot(122)
#plt.imshow(gt)
#plt.show()


print('Desired ratio: ' + str(out_ratio))

print('Original box:')
print('top_left: ' + str(top_left))
print('bot_right: ' + str(bot_right))

if(ratio != out_ratio):
    print('Provided box not compat. with desired ratio.')
    center_box = np.asarray([np.mean([top_left[0],bot_right[0]]),
                  np.mean([top_left[1],bot_right[1]])]).astype(int)
    if(ratio < out_ratio):
        curr_width = bot_right[1] - top_left[1]
        new_width = curr_width*out_ratio/ratio
        add_width_pixs = int(new_width-curr_width)
        print('Will increase width by ' + str(add_width_pixs) + ' pixels')
        new_top_left = [top_left[0],int(top_left[1]-add_width_pixs/2)]
        new_bot_right = [bot_right[0],int(bot_right[1]+add_width_pixs/2)]
    else:
        curr_height = bot_right[0] - top_left[0]
        new_height = curr_height*out_ratio/ratio
        add_height_pixs = int(new_height-curr_height)
        print('Will increase height by ' + str(add_height_pixs) + ' pixels')
        #new_top_left = [top_left[0],int(top_left[1]-add_width_pixs/2)]
        #new_bot_right = [bot_right[0],int(bot_right[1]+add_width_pixs/2)]
        new_top_left = [int(top_left[0]-add_height_pixs/2),top_left[1]]
        new_bot_right = [int(bot_right[0]-add_height_pixs/2),bot_right[1]]

top_left = new_top_left
bot_right = new_bot_right
ratio = (top_left[1]-bot_right[1])/(top_left[0]-bot_right[0])

print('Corrected box:')
print('top_left: ' + str(top_left))
print('bot_right: ' + str(bot_right))
print('ratio: ' + str(ratio))


#Make output dirs
out_dir_ims = os.path.join(dir_root,dir_dataset,'input-frames_resized')
out_dir_gts = os.path.join(dir_root,dir_dataset,'ground_truth-frames_resized')
if(not os.path.exists(out_dir_ims)):
    os.mkdir(out_dir_ims)
else:
    print('Deleting ' + out_dir_ims)
    shutil.rmtree(out_dir_ims)

if(not os.path.exists(out_dir_gts)):
    os.mkdir(out_dir_gts)
else:
    print('Deleting ' + out_dir_gts)
    shutil.rmtree(out_dir_gts)

ims_cropped = []
gts_cropped = []
print('Cropping:')
with progressbar.ProgressBar(maxval=len(ims)) as bar:
    for i in range(len(ims)):
        bar.update(i)
        im = utls.imread(ims[i])
        gt = utls.imread(gts[i])

        new_im = im[top_left[0]:bot_right[0],top_left[1]:bot_right[1],:]
        new_gt = gt[top_left[0]:bot_right[0],top_left[1]:bot_right[1],:]
        ims_cropped.append(new_im)
        gts_cropped.append(new_gt)

ims_resized = []
gts_resized = []
print('Resizing:')
with progressbar.ProgressBar(maxval=len(ims_cropped)) as bar:
    for i in range(len(ims_cropped)):
        bar.update(i)

        new_im = resize(ims_cropped[i],out_size)
        new_gt = resize(gts_cropped[i],out_size)
        ims_resized.append(new_im)
        gts_resized.append(new_gt)

print('Saving:')
with progressbar.ProgressBar(maxval=len(ims_resized)) as bar:
    for i in range(len(ims_resized)):
        bar.update(i)
        imsave(os.path.join(out_dir_ims,os.path.split(ims[i])[-1]),ims_resized[i])
        imsave(os.path.join(out_dir_gts,os.path.split(gts[i])[-1]),gts_resized[i])
