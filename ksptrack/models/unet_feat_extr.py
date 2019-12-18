import torch
import torch.nn as nn
import torch.optim as optim
from ksptrack.models.unet_model import UNet
from ksptrack.models.unet_model import conv1x1
from ksptrack.models.loss_logger import LossLogger
from torch.nn import MSELoss
from ksptrack.models.losses import PriorMSE
from ksptrack.models import utils as ptu
from ksptrack.models import im_utils as ptimu
import time
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
from skimage import (io, transform)
import numpy as np
import tqdm


class UNetFeatExtr(nn.Module):
    """ 
    Wrapper around UNet that takes object priors (gaussians) and images 
    as input.
    """
    def __init__(self,
                 params,
                 in_channels,
                 train_mode,
                 dataloader,
                 depth=3,
                 checkpoint_path=None):

        super(UNetFeatExtr, self).__init__()

        assert ((train_mode == 'autoenc') or
                (train_mode == 'pred')), print('mode must be autoenc or pred')
        if (train_mode == 'pred'):
            assert (checkpoint_path is
                    not None), print('in pred mode, specify checkpoint_path')

        self.dataloader = dataloader
        self.train_mode = train_mode

        self.in_channels = in_channels
        self.out_channels = 3
        self.depth = depth
        self.model = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            merge_mode='none',  # no concatenation (or addition from enc to dec)
            up_mode='upsample',
            depth=depth)

        self.params = params
        self.device = torch.device('cuda' if params['cuda'] else 'cpu')
        self.criterion = PriorMSE(params['cuda'])

        if(train_mode == 'pred'):
            
            print('mode: {}, loading checkpoint {}'.format(train_mode, checkpoint_path))
            dict_ = torch.load(checkpoint_path,
                                map_location=lambda storage,
                                loc: storage)
            self.model.load_state_dict(dict_['model'])
            # replace last conv layer
            self.model.conv_final = conv1x1(self.model.up_convs[-1].out_channels,
                                            1)
            self.criterion = MSELoss()

        self.model.to(self.device)

    def state_dict(self):
        return {
            'params': self.params,
            'depth': self.depth,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'model': self.model.state_dict()
        }

    def get_features_layer(self, x):

        # encoder pathway, omit last downconv block
        for i, module in enumerate(self.model.down_convs):
            x, before_pool = module(x)

        return x

    def calc_features(self, dataloader):

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

    def train(self):

        since = time.time()

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['lr'],
                               betas=(self.params['beta1'],
                                      self.params['beta2']),
                               eps=self.params['epsilon'],
                               weight_decay=self.params['weight_decay_adam'])

        # Save augmented previews
        data_prev = [self.dataloader.dataset.sample_uniform() for i in range(10)]
        path_ = pjoin(self.params['out_dir'], 'augment_previews')
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

        best_loss = np.inf

        for epoch in range(self.params['num_epochs']):
            print('Epoch {}/{}'.format(epoch + 1, self.params['num_epochs']))

            # Each epoch has a training and validation phase
            self.model.train()

            running_loss = 0.0

            pbar = tqdm.tqdm(total=len(self.dataloader))
            for i, sample in enumerate(self.dataloader):

                im = sample['image'].to(self.device)
                truth = sample['label/segmentation'].to(self.device)
                prior = sample['prior'].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # out = self.forward(data.image, data.obj_prior)
                    out = self.model(im)
                    if (self.train_mode == 'autoenc'):
                        loss = self.criterion(out, im, prior)
                    else:
                        loss = self.criterion(out, truth)

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                running_loss += loss.cpu().detach().numpy()
                loss_ = running_loss / ((i + 1) * self.params['batch_size'])
                pbar.set_description('loss: {:.8f}'.format(loss_))
                pbar.update(1)

            pbar.close()

            ptu.save_checkpoint(self.state_dict(),
                                running_loss < best_loss,
                                self.params['out_dir'],
                                fname_cp=self.params['cp_fname'],
                                fname_bm=self.params['bm_fname'])

            if (running_loss < best_loss):
                best_loss = running_loss

            if(out.shape[1] == 1):
                out = out.repeat(1, 3, 1, 1)
            ptimu.save_tensors([im[0], out[0]], ['image', 'image'],
                               os.path.join(self.params['out_dir'],
                                            'train_previews',
                                            'im_{:04d}.png'.format(epoch)))
