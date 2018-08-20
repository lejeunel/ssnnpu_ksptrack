import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import my_utils as utls
import gazeCsv as gaze
from skimage import (color, io, segmentation, morphology)
from skimage.filters.rank import median
from sklearn import (mixture, metrics, preprocessing, decomposition)
from scipy import (ndimage, io)
import logging
from skimage.transform import rescale
import progressbar
from skimage.util import pad
from multiprocessing import Pool
import h5py
from learning_dataset import LearningDataset


class DatasetVilar(LearningDataset):

    def __init__(self, conf):
        LearningDataset.__init__(self,conf)
        self.conf = conf  #Config Bunch object
        self.seen_patches_df = None
        self.unseen_patches_df = None
        self.labels = None

        self.logger = logging.getLogger('DatasetVilar')

        if (not os.path.exists(self.conf.precomp_desc_path)):
            self.logger.info('Feature directory does not exist... creating')
            os.mkdir(self.conf.precomp_desc_path)

    def get_centers_negatives_sps(self, label_pos, labels ):
        """
        label_pos is excluded. All others are returned
        """

        neg_centers = []
        for l in np.unique(labels):
            if(l != label_pos):
                mask = labels == l
                ci, cj = ndimage.measurements.center_of_mass(mask)
                neg_centers.append((int(ci),int(cj)))

        return np.asarray(neg_centers)


    def get_centers_negatives(self, ci, cj, ps, overlap_ratio, shape):

        i_ = []
        j_ = []

        ci = int(ci)
        cj = int(cj)

        if(ps%2): #patch size is odd
            stride = (ps-1)/2
        else:
            stride = ps/2

        # i-dimension up direction
        this_i = int(ci + stride)
        while (this_i < shape[0] - stride):
            i_.append(int(this_i))
            this_i += stride

        # i-dimension down direction
        this_i = int(ci - stride)
        while (this_i > stride):
            i_.append(int(this_i))
            this_i -= stride

        i_.append(ci)

        # j-dimension right direction
        this_j = cj + stride
        while (this_j < shape[1] - stride):
            j_.append(int(this_j))
            this_j += stride

        # j-dimension left direction
        this_j = cj - stride
        while (this_j > stride):
            j_.append(int(this_j))
            this_j -= stride

        j_.append(cj)

        # Remove overlapping boxes
        seen = np.zeros(shape)
        seen[int(ci-stride):int(ci+stride),int(cj-stride):int(cj+stride)] = 1

        new_i = list(i_)
        new_j = list(j_)

        test = np.zeros(shape)
        g_i, g_j = np.meshgrid(i_,j_, indexing='ij')
        test[g_i, g_j] = 1

        pts = np.concatenate((g_i.reshape(-1,1),g_j.reshape(-1,1)),axis=1)
        new_pts = pts.tolist()

        for p in range(pts.shape[0]):
            box = np.zeros(shape)
            box[int(pts[p,0]-stride):int(pts[p,0]+stride),
                        int(pts[p,1]-stride):int(pts[p,1]+stride)] = 1
            if(np.any(np.logical_and(box,seen).ravel()) or
               (np.sum(box) != ps**2)):
                new_pts = [q for q in new_pts
                           if(not((q[0] == pts[p,0]) and (q[1] == pts[p,1])))]


        new_pts = np.asarray(new_pts)

        return new_pts[:,0].astype(int), new_pts[:,1].astype(int)


    def calc_training_patches(self, save=False):

        ps = self.conf.patch_size
        my_gaze = self.conf.myGaze_fg
        scale_factor = self.conf.scale_factor
        seen_patches = list()
        unseen_patches = list()
        selem = morphology.square(5)

        print('Getting seen patches')

        with progressbar.ProgressBar(maxval=len(self.conf.frameFileNames)) as bar:
            for i in range(len(self.conf.frameFileNames)):
                bar.update(i)

                img = utls.imread(self.conf.frameFileNames[i])
                img = (color.rgb2gray(img)*255).astype(np.uint8)
                img = median(img, selem=selem)  # Uses square of size 3

                ci_seen, cj_seen = gaze.gazeCoord2Pixel(
                    my_gaze[i, 3], my_gaze[i, 4], img.shape[1], img.shape[0])

                i_, j_ = self.get_centers_negatives(ci_seen, cj_seen, ps,
                                                    self.conf.overlap_ratio,
                                                    img.shape)

                img_padded = pad(img,((ps,ps),),mode='symmetric')
                seen_patch = img_padded[int(ci_seen + ps/2):int(ci_seen + 3*ps/2),int(cj_seen + ps/2):int(cj_seen + 3*ps/2)]

                seen_patch_mean = np.mean(seen_patch)
                seen_patch_std = np.std(seen_patch)
                if(seen_patch_std == 0): seen_patch_std = 1
                seen_patch = (seen_patch - seen_patch_mean) / seen_patch_std
                seen_patch = rescale(
                    seen_patch,
                    scale=scale_factor,
                    order=1,
                    mode='reflect',
                    preserve_range=True)
                seen_patches.append((i,
                                     ci_seen,
                                     cj_seen,
                                     seen_patch.ravel()))

                for k in range(i_.shape[0]):
                    unseen_patch = img[int(i_[k]- ps / 2):int(i_[k] + ps / 2), int(
                        j_[k]- ps / 2):int(j_[k]+ ps / 2)]
                    unseen_patch_mean = np.mean(unseen_patch)
                    unseen_patch_std = np.std(unseen_patch)
                    if(unseen_patch_std == 0): unseen_patch_std = 1
                    unseen_patch = (unseen_patch - unseen_patch_mean) / unseen_patch_std
                    unseen_patch = rescale(
                        unseen_patch,
                        scale=scale_factor,
                        order=1,
                        mode='reflect',
                        preserve_range=True)
                    unseen_patches.append((i,
                                            i_[k],
                                            j_[k],
                                            unseen_patch.ravel()))
        if(save):
            seen_patches_df = pd.DataFrame(seen_patches,
                                            columns=['frame',
                                                     'c_i',
                                                     'c_j',
                                                     'patch'])
            unseen_patches_df = pd.DataFrame(unseen_patches,
                                            columns=['frame',
                                                     'c_i',
                                                     'c_j',
                                                     'patch'])
            save_path = os.path.join(self.conf.dataOutDir,
                                     'vilar')
            if(not os.path.exists(save_path)):
                os.mkdir(save_path)
            seen_patches_df.to_pickle(os.path.join(save_path,
                                                   'vilar_seen_patches_df.p'))
            unseen_patches_df.to_pickle(
                os.path.join(save_path,
                            'vilar_unseen_patches_df.p'))

        return True

    def get_pred_patches_path(self, im_idx):

        return os.path.join(self.conf.precomp_desc_path,'vilar',
                             'vilar_im_'+str(im_idx)+'.npz')

    def get_pred_patches(self,im_idx):

        fname = os.path.join(self.conf.precomp_desc_path,'vilar',
                             'vilar_im_'+str(im_idx)+'.npz')

        npzfile = np.load(fname)

        return npzfile['patches']

    def get_pred_frame(self,f,dir_=None):

        fname = 'pred_im_' + str(f) + '.npz'
        if(dir_ is None):
            pred = np.load(os.path.join(self.conf.dataOutDir,fname))['pred']
        else:
            pred = np.load(os.path.join(dir_,fname))['pred']

        # Need to reindex
        return pred


    def load_patches(self):

        if((self.seen_patches_df is None) or (self.unseen_patches_df is None)):
            self.logger.info('loading patches')

            self.seen_patches_df = pd.read_pickle(
                os.path.join(self.conf.dataOutDir,'vilar', 'vilar_seen_patches_df.p'))

            self.unseen_patches_df = pd.read_pickle(
                os.path.join(self.conf.dataOutDir,'vilar', 'vilar_unseen_patches_df.p'))

        return self.seen_patches_df, self.unseen_patches_df

    def load_labels_if_not_exist(self):

        if(self.labels is None):

            self.labels = np.load(os.path.join(self.conf.dataInRoot,                                      self.conf.dataSetDir,                                           self.conf.frameDir,'sp_labels.npz'))['sp_labels']

        return self.labels
