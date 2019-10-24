import torch
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
from loss_logger import LossLogger
import time
import os
import im_utils as utls
import copy
import matplotlib.pyplot as plt
from losses import WeightedMSE
from skimage import io
import numpy as np


class UNetObjPrior(nn.Module):
    """ 
    Wrapper around UNet that takes object priors (gaussians) and images 
    as input.
    """

    def __init__(self, params, depth=5):
        super(UNetObjPrior, self).__init__()
        self.in_channels = 4
        self.model = UNet(1, self.in_channels, depth, cuda=params['cuda'])
        self.params = params
        self.device = torch.device('cuda' if params['cuda'] else 'cpu')

    def forward(self, im, obj_prior):
        x = torch.cat((im, obj_prior), dim=1)
        return self.model(x)

    def train(self, dataloader_train, dataloader_val):

        since = time.time()
        best_loss = float("inf")

        dataloader_train.mode = 'train'
        dataloader_val.mode = 'val'
        dataloaders = {'train': dataloader_train, 'val': dataloader_val}

        optimizer = optim.SGD(
            self.model.parameters(),
            momentum=self.params['momentum'],
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay'])

        train_logger = LossLogger('train', self.params['batch_size'],
                                  len(dataloader_train),
                                  self.params['out_dir'])

        val_logger = LossLogger('val', self.params['batch_size'],
                                len(dataloader_val), self.params['out_dir'])

        loggers = {'train': train_logger, 'val': val_logger}

        # self.criterion = WeightedMSE(dataloader_train.get_classes_weights(),
        #                              cuda=self.params['cuda'])
        self.criterion = nn.MSELoss()

        for epoch in range(self.params['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.params['num_epochs'] - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    #scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                samp = 1
                for i, data in enumerate(dataloaders[phase]):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        out = self.forward(data.image, data.obj_prior)
                        loss = self.criterion(out, data.truth)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    loggers[phase].update(epoch, samp, loss.item())

                    samp += 1

                loggers[phase].print_epoch(epoch)

                # Generate train prediction for check
                if phase == 'train':
                    path = os.path.join(self.params['out_dir'], 'previews',
                                        'epoch_{:04d}.jpg'.format(epoch))
                    data = dataloaders['val'].sample_uniform()
                    pred = self.forward(data.image, data.obj_prior)
                    im_ = data.image[0]
                    truth_ = data.truth[0]
                    pred_ = pred[0, ...]
                    utls.save_tensors(im_, pred_, truth_, path)

                if phase == 'val' and (loggers['val'].get_loss(epoch) <
                                       best_loss):
                    best_loss = loggers['val'].get_loss(epoch)

                loggers[phase].save('log_{}.csv'.format(phase))

                # save checkpoint
                if phase == 'val':
                    is_best = loggers['val'].get_loss(epoch) <= best_loss
                    path = os.path.join(self.params['out_dir'],
                                        'checkpoint.pth.tar')
                    utls.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()
                        },
                        is_best,
                        path=path)

    def load_checkpoint(self, path, device='gpu'):

        if (device != 'gpu'):
            checkpoint = torch.load(
                path, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['state_dict'])
