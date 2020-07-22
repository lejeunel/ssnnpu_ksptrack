import os
from os.path import join as pjoin
import pandas as pd
import numpy as np
from ksptrack.utils import superpixel_utils as sputls
from ksptrack.models import im_utils as ptimu
from ksptrack.utils import bagging as bag
from ksptrack.utils.base_dataset import BaseDataset
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from ksptrack.models.losses import PriorMSE
from ksptrack.models import utils as ptu
from skimage import (color, io, segmentation, transform)
import logging
from imgaug import augmenters as iaa
from ksptrack.utils.my_augmenters import rescale_augmenter, Normalize
import torch
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import time
import copy


def get_features(model, cfg, dataloader, checkpoint, mode='autoenc'):

    if (os.path.exists(checkpoint)):
        print('loading {}'.format(checkpoint))
        dict_ = torch.load(checkpoint,
                           map_location=lambda storage, loc: storage)
        model.load_state_dict(dict_)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model.eval()
    model.to(device)

    feats = []

    pbar = tqdm.tqdm(total=len(dataloader))
    for i, sample in enumerate(dataloader):
        with torch.no_grad():
            inputs = sample['image'].to(device)
            res = model(inputs)
            feat = res['feats'].detach().cpu().numpy()
            sp_labels = np.array(
                [s[0, ...].cpu().numpy() for s in sample['labels']])[0]
            feat = [
                feat[0, :, sp_labels == l].mean(axis=0)
                for l in np.unique(sp_labels)
            ]
            feats += feat
        pbar.update(1)

    pbar.close()

    return feats


def train_model(model,
                cfg,
                dataloader,
                checkpoint,
                out_dir,
                cp_fname,
                bm_fname,
                mode='autoenc'):
    since = time.time()

    if (os.path.exists(checkpoint)):
        print('checkpoint {} exists'.format(checkpoint))
        dict_ = torch.load(checkpoint,
                           map_location=lambda storage, loc: storage)
        model.load_state_dict(dict_)

        return model

    model.to_autoenc()
    # criterion = PriorMSE()
    criterion = torch.nn.MSELoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model.to(device)
    optimizer = optim.SGD(params=[{
        'params': model.parameters(),
        'lr': cfg.feat_sgd_learning_rate
    }],
                          momentum=cfg.feat_sgd_momentum,
                          weight_decay=cfg.feat_sgd_decay)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, cfg.feat_sgd_learning_rate_power)

    # Save augmented previews
    data_prev = [dataloader.dataset.sample_uniform() for i in range(10)]
    path_ = pjoin(out_dir, 'augment_previews_{}'.format(mode))
    if (not os.path.exists(path_)):
        os.makedirs(path_)

    padding = ((10, ), (10, ), (0, ))
    data_prev = [
        np.concatenate(
            (np.pad(d['image'], pad_width=padding, mode='constant'),
             np.pad(d['prior'], pad_width=padding, mode='constant')),
            axis=-1) for d in data_prev
    ]

    for i, d in enumerate(data_prev):
        io.imsave(pjoin(path_, 'im_{:02d}.png'.format(i)), d)

    max_itr = cfg.feat_n_epochs * len(dataloader)
    itr = 0
    for epoch in range(cfg.feat_n_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.feat_n_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0

        # Iterate over data.
        pbar = tqdm.tqdm(total=len(dataloader))
        for sample in dataloader:
            input = sample['image'].to(device)
            prior = sample['prior'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                res = model(input)
                # loss = criterion(res['output'], input, prior)
                loss = criterion(res['output'], input)

                loss.backward()
                optimizer.step()
                itr += 1

            # statistics
            running_loss += loss.item() * input.size(0)
            pbar.set_description('loss: {:.8f}'.format(running_loss))
            pbar.update(1)

        pbar.close()
        lr_sch.step()

        epoch_loss = running_loss / len(dataloader.dataset)

        print('{} Loss: {:.4f} LR: {}'.format('train', epoch_loss,
                                              lr_sch.get_lr()[0]))

        if (running_loss < best_loss):
            best_loss = running_loss

        output = res['output']
        if (output.shape[1] == 1):
            output = output.repeat(1, 3, 1, 1)
        ptimu.save_tensors([input[0], output[0]], ['image', 'image'],
                           os.path.join(out_dir, 'train_previews',
                                        'im_{:04d}.png'.format(epoch)))

        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            ptu.save_checkpoint(model.state_dict(),
                                running_loss < best_loss,
                                out_dir,
                                fname_cp=cp_fname.format(mode),
                                fname_bm=bm_fname.format(mode))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class DataManager:
    def __init__(self,
                 root_path,
                 desc_dir='precomp_desc',
                 feats_mode='autoenc'):
        self.desc_path = pjoin(root_path, desc_dir)
        self.desc_dir = desc_dir
        self.root_path = root_path
        self.feats_mode = feats_mode

        self.logger = logging.getLogger('Dataset')

        self.labels_ = None
        self.labels_contours_ = None
        self.centroids_loc_ = None
        self.sp_desc_df_ = None

        if (not os.path.exists(self.desc_path)):
            self.logger.info('Feature directory does not exist... creating')
            os.mkdir(self.desc_path)

    @property
    def labels_contours(self):

        if (self.labels_contours_ is None):
            self.labels_contours_ = np.load(
                os.path.join(self.desc_path, 'sp_labels_tsp_contours.npz')
            )['labels_contours'].transpose((1, 2, 0))
        return self.labels_contours_

    @property
    def sp_desc_df(self):
        return self.get_sp_desc_from_file()

    @property
    def centroids_loc(self):

        if (self.centroids_loc_ is None):
            centroid_path = os.path.join(self.desc_path, 'centroids_loc_df.p')
            self.logger.info(
                'loading superpixel centroids: {}'.format(centroid_path))
            self.centroids_loc_ = pd.read_pickle(centroid_path)
        return self.centroids_loc_

    def get_sp_desc_from_file(self):

        fname = 'sp_desc_{}.p'.format(self.feats_mode)

        if (self.sp_desc_df_ is None):
            path = os.path.join(self.desc_path, fname)
            print('loading features at {}'.format(path))
            out = pd.read_pickle(path)
            self.sp_desc_df_ = out

        return self.sp_desc_df_

    def load_pm_fg_from_file(self):
        self.logger.info("Loading PM foreground")
        self.fg_marked = np.load(os.path.join(self.desc_path,
                                              'fg_marked.npz'))['fg_marked']
        self.fg_pm_df = pd.read_pickle(
            os.path.join(self.desc_path, 'fg_pm_df.p'))

    def calc_superpix(self, compactness, n_segments, do_save=True):
        """
        Makes centroids and contours
        """

        if (not os.path.exists(pjoin(self.desc_path, 'sp_labels.npz'))):
            dset = BaseDataset(self.root_path, got_labels=False)

            self.logger.info('Running SLIC on {} images with {} labels'.format(
                len(dset), n_segments))
            labels = np.array([
                segmentation.slic(s['image'],
                                  n_segments=n_segments,
                                  compactness=compactness) for s in dset
            ])
            self.logger.info('Saving labels to {}'.format(self.desc_path))
            np.save(os.path.join(self.desc_path, 'sp_labels.npy'), labels)

            self.labels_contours_ = list()
            self.logger.info("Generating label contour maps")

            for im in range(self.labels.shape[2]):
                # labels values are not always "incremental" (values are skipped).
                self.labels_contours_.append(
                    segmentation.find_boundaries(self.labels[:, :, im]))

            self.labels_contours_ = np.array(self.labels_contours_)
            self.logger.info("Saving labels")
            data = dict()
            data['labels_contours'] = self.labels_contours
            np.savez(os.path.join(self.desc_path, 'sp_labels_contours.npz'),
                     **data)

            if (do_save):
                self.logger.info('Saving slic previews to {}'.format(
                    pjoin(self.desc_path, 'spix_previews')))
                previews_dir = os.path.join(self.desc_path, 'spix_previews')
                if (not os.path.exists(previews_dir)):
                    os.makedirs(previews_dir)
                for i, sample in enumerate(dset):
                    fname = os.path.join(previews_dir,
                                         'frame_{0:04d}.png'.format(i))

                    im = sputls.drawLabelContourMask(sample['image'],
                                                     self.labels[..., i])
                    io.imsave(fname, im)

            self.logger.info('Getting centroids...')
            self.centroids_loc_ = sputls.getLabelCentroids(self.labels)

            self.centroids_loc_.to_pickle(
                os.path.join(self.desc_path, 'centroids_loc_df.p'))
        else:
            self.logger.info(
                "Superpixels were already computed. Delete to re-run.")

    def calc_sp_feats(self, cfg):
        """ 
        Computes UNet features in Autoencoder-mode
        Train/forward-propagate Unet and save features/weights
        """

        df_fname = 'sp_desc_{}.p'
        cp_fname = 'checkpoint_{}.pth.tar'
        bm_fname = 'best_model_{}.pth.tar'

        df_path = os.path.join(self.desc_path, df_fname)
        bm_path = os.path.join(self.desc_path, bm_fname)
        cp_path = os.path.join(self.desc_path, cp_fname)

        from ksptrack.models.deeplab import DeepLabv3Plus

        transf = iaa.Sequential([
            iaa.SomeOf(cfg.feat_data_someof, [
                iaa.Affine(
                    scale={
                        "x": (1 - cfg.feat_data_width_shift,
                              1 + cfg.feat_data_width_shift),
                        "y": (1 - cfg.feat_data_height_shift,
                              1 + cfg.feat_data_height_shift)
                    },
                    rotate=(-cfg.feat_data_rot_range, cfg.feat_data_rot_range),
                    shear=(-cfg.feat_data_shear_range,
                           cfg.feat_data_shear_range)),
                iaa.AdditiveGaussianNoise(
                    scale=cfg.feat_data_gaussian_noise_std * 255),
                iaa.Fliplr(p=0.5),
                iaa.Flipud(p=0.5)
            ]), rescale_augmenter,
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dl = LocPriorDataset(root_path=self.root_path,
                             sig_prior=cfg.feat_locs_gaussian_std,
                             augmentations=transf)
        if (self.feats_mode == 'autoenc'):
            sampler = None
            shuffle = True
        else:
            idx_refine = np.repeat(np.arange(len(dl)), 120 // len(dl))
            sampler = SubsetRandomSampler(idx_refine)
            shuffle = False

        dataloader = DataLoader(dl,
                                batch_size=cfg.batch_size,
                                shuffle=shuffle,
                                sampler=sampler,
                                collate_fn=dl.collate_fn,
                                drop_last=True,
                                num_workers=cfg.feat_n_workers)
        model = DeepLabv3Plus(pretrained=False)
        train_model(model, cfg, dataloader, cp_path.format('autoenc'),
                    self.desc_path, cp_fname, bm_fname, self.feats_mode)

        # Do forward pass on images and retrieve features
        if (not os.path.exists(df_path.format(self.feats_mode))):
            self.logger.info("Computing features on superpixels")

            transf = iaa.Sequential([
                rescale_augmenter,
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])

            dl.augmentations = transf
            dataloader = DataLoader(dl, batch_size=1, collate_fn=dl.collate_fn)
            model = DeepLabv3Plus()
            feats_sp = get_features(model, cfg, dataloader,
                                    cp_path.format(self.feats_mode),
                                    self.feats_mode)

            feats_df = self.centroids_loc.assign(desc=feats_sp)
            self.logger.info('Saving  features to {}.'.format(
                df_path.format(self.feats_mode)))
            feats_df.to_pickle(os.path.join(df_path.format(self.feats_mode)))

    def get_pm_array(self, save=False, frames=None):
        """ Returns array same size as labels with probabilities of bagging model
        """

        scores = self.labels.copy().astype(float)
        pm_df = self.fg_pm_df

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
            # Create 2D replacement matrix
            replace = np.array(
                [list(dict_map.keys()),
                 list(dict_map.values())])
            # Find elements that need replacement
            mask = np.isin(scores[..., f], replace[0, :])
            # Replace elements
            scores[mask,
                   f] = replace[1,
                                np.searchsorted(replace[0, :], scores[mask,
                                                                      f])]
            # for k, v in dict_map.items():
            #     scores[scores[..., f] == k, f] = v
            i += 1
            bar.update(1)
        bar.close()

        return scores

    def calc_svc(self, X, y):

        from ksptrack.utils.puAdapter import PUAdapter
        from sklearn.svm import SVC

        y = y.astype(int)
        y[np.where(y == 0)[0]] = -1.
        estimator = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
        pu_estimator = PUAdapter(estimator, hold_out_ratio=0.2)

        self.logger.info('Fitting PU SVC model')
        pu_estimator.fit(X, y)

        probas = pu_estimator.predict(X)

        self.fg_pm_df = pd.DataFrame({
            'frame': self.sp_desc_df['frame'],
            'label': self.sp_desc_df['label'],
            'proba': probas
        })

        return self.fg_pm_df
