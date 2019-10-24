import os
from os.path import join as pjoin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import superpixel_utils as spix
from ksptrack.utils import my_utils as utls
from ksptrack.utils import bagging as bag
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import superpixel_extractor as svx
from ksptrack.utils import superpixel_utils as spix_utls
from skimage import (color, io, segmentation, transform)
from sklearn import (mixture, metrics, preprocessing, decomposition)
import scipy as scp
import glob, itertools
import logging
from imgaug import augmenters as iaa
from ksptrack.utils.my_augmenters import rescale_augmenter
import torch
import tqdm


class DataManager:
    def __init__(self, conf):
        self.conf = conf  #Config Bunch object

        self.logger = logging.getLogger('Dataset')

        self.labels_ = None
        self.labels_contours_ = None
        self.centroids_loc_ = None
        self.sp_desc_df_ = None

        if (not os.path.exists(self.conf.precomp_desc_path)):
            self.logger.info('Feature directory does not exist... creating')
            os.mkdir(self.conf.precomp_desc_path)

    @property
    def labels_contours(self):

        if (self.labels_contours_ is None):
            self.labels_contours_ = np.load(
                os.path.join(self.conf.precomp_desc_path,
                             'sp_labels_tsp_contours.npz')
            )['labels_contours'].transpose((1, 2, 0))
        return self.labels_contours_

    @property
    def labels(self):

        if (self.labels_ is None):
            path_ = os.path.join(self.conf.precomp_desc_path, 'sp_labels.npz')
            self.logger.info('loading superpixel labels: {}'.format(path_))

            self.labels_ = np.load(path_)['sp_labels']

        return self.labels_

    @property
    def sp_desc_df(self):
        return self.get_sp_desc_from_file()

    @property
    def centroids_loc(self):

        if(self.centroids_loc_ is None):
            centroid_path = os.path.join(self.conf.precomp_desc_path,
                                        'centroids_loc_df.p')
            self.logger.info(
                'loading superpixel centroids: {}'.format(centroid_path))
            self.centroids_loc_ = pd.read_pickle(centroid_path)
        return self.centroids_loc_

    def get_sp_desc_from_file(self):

        fname = 'sp_desc_ung.p'

        if(self.sp_desc_df_ is None):
            path = os.path.join(self.conf.precomp_desc_path, fname)
            out = pd.read_pickle(path)
            self.sp_desc_df_ = out

        return self.sp_desc_df_

    def get_sp_desc_means_from_file(self):

        path = os.path.join(self.conf.precomp_desc_path, 'sp_desc_means.p')
        if (os.path.exists(path)):
            out = pd.read_pickle(os.path.join(path))
        else:
            return None

        return out

    def load_sp_desc_means_from_file(self):
        self.logger.info("Loading sp means ")
        self.sp_desc_means = pd.read_pickle(
            os.path.join(self.conf.precomp_desc_path, 'sp_desc_means.p'))

    def load_pm_all_feats(self):
        self.logger.info("Loading PM all feats")
        self.all_feats_df = pd.read_pickle(
            os.path.join(self.conf.precomp_desc_path, 'all_feats_df.p'))

    def load_pm_fg_from_file(self):
        self.logger.info("Loading PM foreground")
        self.fg_marked = np.load(
            os.path.join(self.conf.precomp_desc_path,
                         'fg_marked.npz'))['fg_marked']
        self.fg_pm_df = pd.read_pickle(
            os.path.join(self.conf.precomp_desc_path, 'fg_pm_df.p'))
        self.fg_marked_feats = np.load(
            os.path.join(self.conf.precomp_desc_path,
                         'fg_marked_feats.npz'))['fg_marked_feats']

    def calc_superpix(self, do_save=True):
        """
        Makes centroids and contours
        """

        if(not os.path.exists(pjoin(self.conf.precomp_desc_path, 'sp_labels.npz'))):
            spix_extr = svx.SuperpixelExtractor()

            self.labels_, numlabels = spix_extr.extract(
                self.conf.frameFileNames,
                self.conf.precomp_desc_path,
                'sp_labels.npz',
                self.conf.slic_compactness,
                self.conf.slic_n_sp)

            self.labels_contours_ = list()
            self.logger.info("Generating label contour maps")

            for im in range(self.labels.shape[2]):
                # labels values are not always "incremental" (values are skipped).
                self.labels_contours_.append(
                    segmentation.find_boundaries(self.labels[:, :, im]))

            self.labels_contours_ = np.array(self.labels_contours_)
            self.logger.info("Saving label contour maps")
            data = dict()
            data['labels_contours'] = self.labels_contours
            np.savez(
                os.path.join(self.conf.precomp_desc_path,
                            'sp_labels_tsp_contours.npz'), **data)

            if (do_save):
                self.logger.info('Saving labels to {}'.format(
                    self.conf.precomp_desc_path))
                np.savez(
                    os.path.join(self.conf.precomp_desc_path, 'sp_labels.npz'), **{
                        'sp_labels': self.labels,
                        'numlabels': numlabels
                    })

            if (do_save):
                self.logger.info('Saving slic previews to {}'.format(
                    pjoin(self.conf.precomp_desc_path, 'spix_previews')))
                ims_list = [utls.imread(im) for im in self.conf.frameFileNames]
                previews_dir = os.path.join(self.conf.precomp_desc_path,
                                            'spix_previews')
                if (not os.path.exists(previews_dir)):
                    os.makedirs(previews_dir)
                for i, im in enumerate(ims_list):
                    fname = os.path.join(previews_dir,
                                        'frame_{0:04d}.png'.format(i))

                    im = spix_utls.drawLabelContourMask(im, self.labels[..., i])
                    io.imsave(fname, im)

            self.logger.info('Getting centroids...')
            self.centroids_loc_ = spix.getLabelCentroids(self.labels)

            self.centroids_loc_.to_pickle(
                os.path.join(self.conf.precomp_desc_path,
                            'centroids_loc_df.p'))
        else:
            self.logger.info("Superpixels were already computer. Delete to re-run.")

    def calc_bagging(self,
                     marked_arr,
                     all_feats_df=None,
                     mode='foreground',
                     T=100,
                     bag_n_feats=0.25,
                     bag_max_depth=5,
                     bag_max_samples=500,
                     n_jobs=1):
        """
        Computes "Bagging" transductive probabilities using marked_arr as positives.
        """

        this_marked_feats, this_pm_df = bag.calc_bagging(
            T,
            bag_max_depth,
            bag_n_feats,
            marked_arr,
            all_feats_df=all_feats_df,
            feat_fields=['desc'],
            bag_max_samples=bag_max_samples,
            n_jobs=n_jobs)

        if (mode == 'foreground'):
            self.fg_marked_feats = this_marked_feats
            self.fg_pm_df = this_pm_df
        else:
            self.bg_marked_feats = this_marked_feats
            self.bg_pm_df = this_pm_df

        return self.fg_pm_df

    def calc_sp_feats_unet_gaze_rec(self, locs2d, save_dir=None):
        """ 
        Computes UNet features in Autoencoder-mode
        Train/forward-propagate Unet and save features/weights
        """

        df_fname = 'sp_desc_ung.p'
        feats_fname = 'feats_unet.npz'
        feats_upsamp_fname = 'feats_unet_upsamp.npz'
        cp_fname = 'checkpoint.pth.tar'
        bm_fname = 'best_model.pth.tar'

        feats_path = os.path.join(save_dir, feats_fname)
        feats_upsamp_path = os.path.join(save_dir, feats_upsamp_fname)
        df_path = os.path.join(save_dir, df_fname)
        bm_path = os.path.join(save_dir, bm_fname)

        from ksptrack.models.unet_feat_extr import UNetFeatExtr
        from ksptrack.models.dataset import Dataset

        orig_shape = utls.imread(self.conf.frameFileNames[50]).shape
        in_shape = self.conf.feat_in_shape

        params = {
            'batch_size': self.conf.batch_size,
            'cuda': self.conf.cuda,
            'lr': self.conf.feat_adam_learning_rate,
            'momentum': 0.9,
            'beta1': self.conf.feat_adam_beta1,
            'beta2': self.conf.feat_adam_beta2,
            'epsilon': self.conf.feat_adam_epsilon,
            'weight_decay_adam': self.conf.feat_adam_decay,
            'num_epochs': self.conf.feat_n_epochs,
            'out_dir': save_dir,
            'cp_fname': cp_fname,
            'bm_fname': bm_fname,
            'n_workers': self.conf.feat_n_workers
        }

        train_net = not os.path.exists(bm_path)
        forward_pass = not os.path.exists(feats_path)
        assign_feats_to_sps = not os.path.exists(df_path)

        if (train_net):
            self.logger.info(
                "Network weights file {} does not exist. Training...".format(
                    bm_path))
            self.logger.info(
                "Parameters: \n {}".format(
                    params))

            model = UNetFeatExtr(params)
            model.model.train()

            transf = iaa.Sequential([
                iaa.SomeOf(self.conf.feat_data_someof,
                           [iaa.Affine(
                               scale={
                                   "x": (1 - self.conf.feat_data_width_shift,
                                         1 + self.conf.feat_data_width_shift),
                                   "y": (1 - self.conf.feat_data_height_shift,
                                         1 + self.conf.feat_data_height_shift)
                               },
                               rotate=(-self.conf.feat_data_rot_range,
                                       self.conf.feat_data_rot_range),
                               shear=(-self.conf.feat_data_shear_range,
                                      self.conf.feat_data_shear_range)),
                            iaa.AdditiveGaussianNoise(
                                scale=self.conf.feat_data_gaussian_noise_std*255),
                            iaa.Fliplr(p=0.5),
                            iaa.Flipud(p=0.5)]),
                iaa.Resize(self.conf.feat_in_shape),
                rescale_augmenter])

            dl = Dataset(
                in_shape,
                im_paths=self.conf.frameFileNames,
                truth_paths=None,
                locs2d=locs2d,
                sig_prior=self.conf.feat_locs_gaussian_std,
                augmentations=transf,
                cuda=self.conf.cuda)

            model.train(dl)

        # Do forward pass on images and retrieve features
        if (forward_pass):
            # load best model
            dict_ = torch.load(bm_path,
                               map_location=lambda storage,
                               loc: storage)
            model = UNetFeatExtr.from_state_dict(dict_)

            device = torch.device('cuda' if self.conf.cuda else 'cpu')
            model.to(device)

            transf = iaa.Sequential([
                iaa.Resize(self.conf.feat_in_shape),
                rescale_augmenter])

            dl = Dataset(
                in_shape,
                im_paths=self.conf.frameFileNames,
                augmentations=transf,
                cuda=self.conf.cuda)

            if(not os.path.exists(feats_path)):
                self.logger.info(
                    "Downsampled feature file {} does not exist...".format(
                        feats_path))

                model.model.eval()
                feats_ds = model.calc_features(dl)

                self.logger.info(
                    'Saving (downsampled) features to {}.'.format(feats_path))
                np.savez(feats_path, **{'feats': feats_ds})
        else:
            self.logger.info(
                "Downsampled feature file {} exist. Delete to re-run.".format(
                    feats_path))
            feats_ds = np.load(feats_path)['feats']

        if(assign_feats_to_sps):
            self.logger.info("Computing features on superpixels")
            feats_sp = list()
            frames = np.unique(self.centroids_loc['frame'].values)
            bar = tqdm.tqdm(total=len(frames))
            for f in frames:
                feats_us = transform.resize(feats_ds[..., f],
                                            (*orig_shape[0:2], feats_ds.shape[2]))
                            
                for index, row in self.centroids_loc[
                        self.centroids_loc['frame'] == f].iterrows():
                    x = row['x']
                    y = row['y']
                    ci, cj = csv.coord2Pixel(x, y, feats_us.shape[1],
                                                feats_us.shape[0])
                    feats_sp.append(feats_us[ci, cj, :].copy())
                bar.update(1)
            bar.close()

            feats_df = self.centroids_loc.assign(desc=feats_sp)
            self.logger.info('Saving  features to {}.'.format(df_path))
            feats_df.to_pickle(os.path.join(df_path))
        else:
            self.logger.info(
                "feature file {} exist. Delete to re-run.".format(
                    df_path))
            feats_df = pd.read_pickle(df_path)


    def get_pm_array(self, mode='foreground', save=False, frames=None):
        """ Returns array same size as labels with probabilities of bagging model
        """

        scores = self.labels.copy().astype(float)
        if (mode == 'foreground'):
            pm_df = self.fg_pm_df
        else:
            pm_df = self.bg_pm_df

        if (frames is None):  #Make all frames
            frames = np.arange(scores.shape[-1])
        else:
            frames = np.asarray(frames)

        i = 0
        self.logger.info('Generating PM array')
        bar = tqdm.tqdm(total=len(frames))
        for f in frames:
            this_frame_pm_df = pm_df[pm_df['frame'] == f]
            dict_keys = this_frame_pm_df['label']
            dict_vals = this_frame_pm_df['proba']
            dict_map = dict(zip(dict_keys, dict_vals))
            for k, v in dict_map.items():
                scores[scores[..., f] == k, f] = v
            i += 1
            bar.update(1)
        bar.close()

        if (save):

            if (mode == 'foreground'):
                data = dict()
                data['pm_scores'] = scores
                np.savez(
                    os.path.join(self.conf.precomp_desc_path,
                                 'pm_scores_fg_orig.npz'), **data)
            else:
                data = dict()
                data['pm_scores'] = scores
                np.savez(
                    os.path.join(self.conf.precomp_desc_path,
                                 'pm_scores_fg_orig.npz'), **data)

        return scores

    def calc_pm(self,
                pos_sps,
                all_feats_df=None,
                mode='foreground'):
        """
        Main function that extracts or updates gaze coordinates
        and computes transductive learning model (bagging)
        Inputs:
            mode: {'foreground','background'}
        """

        self.logger.info('--- Generating probability map foreground')
        self.logger.info("T: " + str(self.conf.bag_t))
        self.logger.info("bag_n_bins: " + str(self.conf.bag_n_bins))


        self.calc_bagging(
            pos_sps,
            all_feats_df=all_feats_df,
            mode=mode,
            T=self.conf.bag_t,
            bag_n_feats=self.conf.bag_n_feats,
            bag_max_depth=self.conf.bag_max_depth,
            bag_max_samples=self.conf.bag_max_samples,
            n_jobs=self.conf.bag_jobs)
