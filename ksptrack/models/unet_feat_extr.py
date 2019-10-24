import torch
import torch.nn as nn
import torch.optim as optim
from ksptrack.models.unet_model import UNet
from ksptrack.models.loss_logger import LossLogger
from ksptrack.models import utils as ptu
from ksptrack.models import im_utils as ptimu
from ksptrack.models.losses import PriorMSE
import time
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
from skimage import (io, transform)
import numpy as np
from itertools import cycle
from torch.utils.data import DataLoader
import tqdm
from imgaug import augmenters as iaa
from imgaug import parameters as iap

class UNetFeatExtr(nn.Module):
    """ 
    Wrapper around UNet that takes object priors (gaussians) and images 
    as input.
    """

    def __init__(self, params, depth=4):
        super(UNetFeatExtr, self).__init__()
        self.in_channels = 3
        self.out_channels = 3
        self.depth = depth
        self.model = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            merge_mode=
            'none',  # no concatenation (or addition from enc to dec)
            up_mode='upsample',
            depth=depth,
            cuda=params['cuda'])
        self.params = params
        self.device = torch.device('cuda' if params['cuda'] else 'cpu')

    def state_dict(self):
        return {'params': self.params,
                'depth': self.depth,
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'model': self.model.state_dict()}

    @classmethod
    def from_state_dict(cls, dict_):
        feat_extr = cls(dict_['params'], depth=dict_['depth'])
        feat_extr.model.load_state_dict(dict_['model'])
        return feat_extr

    def forward(self, x):
        return self.model(x)

    def get_features_layer(self, x):

        # encoder pathway, omit last downconv block
        for i, module in enumerate(self.model.down_convs):
            x, before_pool = module(x)

        return x

    def calc_features(self, dataset):

        dataloader = DataLoader(dataset,
                                batch_size=self.params['batch_size'],
                                shuffle=False,
                                collate_fn=dataset.collate_fn,
                                num_workers=self.params['n_workers'])

        feats = []
        bar = tqdm.tqdm(total=len(dataloader))
        for i, sample in enumerate(dataloader):
            im = sample['image'].to(self.device)
            f_ = self.get_features_layer(im)
            feats.append(f_.detach().cpu().numpy())
            bar.update(1)

        bar.close()

        feats = np.concatenate(feats)
        feats = feats.transpose((2, 3, 1, 0))
        return feats

    def train(self, dataset):

        since = time.time()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['lr'],
            betas=(self.params['beta1'], self.params['beta2']),
            eps=self.params['epsilon'],
            weight_decay=self.params['weight_decay_adam'])

        train_logger = LossLogger(
            'train',
            self.params['batch_size'],
            len(dataset),
            self.params['out_dir'],
            print_mode=0)

        loggers = {'train': train_logger}

        dataloader = DataLoader(dataset,
                                batch_size=self.params['batch_size'],
                                shuffle=True,
                                collate_fn=dataset.collate_fn,
                                num_workers=self.params['n_workers'])

        # Save augmented previews
        data_prev = [dataset.sample_uniform() for i in range(10)]
        path_ = pjoin(self.params['out_dir'], 'augment_previews')
        if(not os.path.exists(path_)):
            os.makedirs(path_)

        padding = ((10,), (10,), (0,))
        data_prev = [np.concatenate((np.pad(d['image'],
                                            pad_width=padding,
                                            mode='constant'),
                                     np.pad(d['prior'],
                                            pad_width=padding,
                                            mode='constant')),
                                    axis=-1)
                     for d in data_prev]
        for i, d in enumerate(data_prev):
            io.imsave(pjoin(path_,
                            'im_{:02d}.png'.format(i)),
            d)

        # Path for checkpoint
        self.criterion = PriorMSE(self.params['cuda'])

        best_loss = np.inf

        for epoch in range(self.params['num_epochs']):
            print('Epoch {}/{}'.format(epoch + 1, self.params['num_epochs']))

            # Each epoch has a training and validation phase
            self.model.train()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            samp = 1
            n_batches_per_epoch = len(dataset) // self.params['batch_size']

            running_loss = 0.0

            pbar = tqdm.tqdm(total=len(dataloader))
            for i, sample in enumerate(dataloader):

                im = sample['image'].to(self.device)
                prior = sample['prior'].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # out = self.forward(data.image, data.obj_prior)
                    out = self.forward(im)
                    loss = self.criterion(out, im, prior)

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                running_loss += loss.cpu().detach().numpy()
                loss_ = running_loss / ((i + 1) * self.params['batch_size'])
                pbar.set_description('loss: {:.8f}'.format(
                    loss_))
                pbar.update(1)
                samp += 1

            pbar.close()

            ptu.save_checkpoint(self.state_dict(),
                                running_loss < best_loss,
                                self.params['out_dir'],
                                fname_cp=self.params['cp_fname'],
                                fname_bm=self.params['bm_fname'])

            if(running_loss < best_loss):
                best_loss = running_loss
                            
            ptimu.save_tensors(
                [im[0], out[0]],
                ['image', 'image'],
                os.path.join(self.params['out_dir'], 'train_previews',
                                'im_{:04d}.png'.format(epoch)))
