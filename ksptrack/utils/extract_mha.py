import SimpleITK as sitk
from medpy import io
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import my_utils as utls
from skimage import color
import numpy.ma as ma
from skimage import transform

im_comp_size = (100, 100)

def comp_features(x, y):

    dists = []

    for i in range(x.shape[-1]):
        for j in range(y.shape[-1]):
            dists.append(np.sum(np.abs(x[..., i] - y[..., j])))

    return dists

def find_T2_dir(path):
    dirs = os.listdir(path)
    has_T2 = ['3more' in d for d in dirs]

    if(np.sum(has_T2) == 0):
        return None
    else:
        return dirs[np.argmax(has_T2)]

def get_mha(path):
    images = sitk.ReadImage(path)
    max_index = images.GetDepth()

    images_np = np.asarray([sitk.GetArrayFromImage(images[:,:,i])
                            for i in range(max_index)]).transpose((1,2,0))

    # Normalize
    images_np = (images_np/np.max(images_np)*255).astype(np.uint8)

    # Make "grayscale"
    out = np.asarray([np.repeat(images_np[..., i, np.newaxis], 3, axis=-1)
                           for i in range(images_np.shape[-1])])

    out = out.transpose((1,2,3,0))

    return out

def inters_mask_labels(labels, mask_, val):

    tmp = np.logical_and(mask_, (labels==val))
    #plt.subplot(121); plt.imshow(mask_)
    #plt.subplot(122); plt.imshow(labels == val)
    #plt.show()

    return np.sum(tmp)/np.sum(labels==val)

def get_feats(ims, n_bins=30, square_mask=True):

    ims_1c = ims[:, :, -1, :]
    mask = np.zeros(ims[...,0].shape[0:2], dtype=np.uint8)

    if(mask.shape[1] >= mask.shape[0]):
        diff = mask.shape[1] - mask.shape[0]
        range_cols = (diff//2, mask.shape[1]-diff//2-1)
        range_rows = (0, mask.shape[0]-1)
    else:
        diff = mask.shape[0] - mask.shape[1]
        range_cols = (0, mask.shape[1]-1)
        range_rows = (diff//2, mask.shape[0]-diff//2-1)

    im_list = list()

    for i in range(ims.shape[-1]):
        im_ = ims_1c[..., i].astype(float)
        im_ = im_[np.meshgrid(np.arange(range_rows[0], range_rows[1]),
                              np.arange(range_cols[0], range_cols[1]),
                              indexing='ij')]
        im_ = transform.resize(im_, im_comp_size)
        im_list.append(im_)

    return im_list

root_dir = '/home/laurent.lejeune/Downloads/BRATS2015_Training'
#root_dir = '/home/laurent.lejeune/Downloads/BRATS-2/Image_Data'

sub_root_dir = ['HGG', 'LGG']
#sub_root_dir = ['HG', 'LG']

my_root_paths = ['/home/laurent.lejeune/medical-labeling/Dataset30/ground_truth-frames',
                 '/home/laurent.lejeune/medical-labeling/Dataset31/ground_truth-frames',
                 '/home/laurent.lejeune/medical-labeling/Dataset32/ground_truth-frames',
                 '/home/laurent.lejeune/medical-labeling/Dataset33/ground_truth-frames']

path_saves = '/home/laurent.lejeune/medical-labeling/'
save_out_fname = 'brats_matching_1.npz'

# Range (normalized) of volume to compute error
range_comp = np.asarray((0, 1.))
im_ind_my = [70, 28, 20, 44]

my_paths = [sorted(glob.glob(os.path.join(p, '*.png'))) for p in my_root_paths]

ims_my = list()
for i in range(len(my_paths)):
    for j in range(len(my_paths[i])):
        if(j == im_ind_my[i]):
            ims_my.append(np.flip(utls.imread(my_paths[i][j]), 1)[..., np.newaxis])

print('Extracting feats on my')
im_my = dict()
for i in range(len(ims_my)):
    im_= get_feats(ims_my[i])
    im_my[my_root_paths[i]] = im_
print('Done.')

print('Saving features on my')
np.savez(os.path.join(path_saves, 'hists_my.npz'),
         **{'ims': im_my})

match_my = dict()

print('Matching on brats')
for sd, si in zip(sub_root_dir, range(len(sub_root_dir))):
    path_ = os.path.join(root_dir, sd)
    all_datasets_paths = os.listdir(path_)

    # Loop over all BRATS datasets
    for d, di in zip(all_datasets_paths, range(len(all_datasets_paths))):
        ratio_sub_root = (si+1)/len(sub_root_dir)
        ratio_dset = (di+1)/len(all_datasets_paths)
        print('{0:.2f}%'.format(ratio_sub_root*ratio_dset*100))
        dset_path = find_T2_dir(os.path.join(path_, d))
        if(dset_path is not None):

            mha_path = glob.glob(os.path.join(path_,
                                    d,
                                    dset_path,
                                    '*.mha'))

            ims_mha = get_mha(mha_path[0])

            # Get frames to compare
            range_mha = (ims_mha.shape[-1]*range_comp).astype(int)
            ims_mha_ = ims_mha[..., range(range_mha[0], range_mha[1])]

            im_comp = get_feats(ims_mha_)
            for k in im_my.keys():
                dist = comp_features(np.asarray(im_my[k]).transpose((1,2,0)),
                                    np.asarray(im_comp).transpose((1,2,0)))
                print('{}, {}: min_dist = {}'.format(k, mha_path, np.min(dist)))
                if(k not in match_my.keys()):
                    match_my[k] = dict()
                match_my[k].update({mha_path[0]: dist})
                #match_my[k][dset_path] =  dist

print('Done.')

print('Saving matching')
np.savez(os.path.join(path_saves, save_out_fname),
         **{'match': match_my})
