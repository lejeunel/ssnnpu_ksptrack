from loader import Loader
from imgaug import augmenters as iaa
import logging
from my_augmenters import rescale_augmenter, Normalize
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
import torch.optim as optim
import params
import torch
import datetime
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
import tqdm
from skimage.future.graph import show_rag
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.models.deeplab_resnet import DeepLabv3_plus
from ksptrack.siamese import im_utils
from skimage import color, io
from torch import functional as F
import numpy as np


class PriorMSELoss(torch.nn.Module):
    def __init__(self):
        super(PriorMSELoss, self).__init__()

    def forward(self, y, y_true, prior):

        L = ((y - y_true).pow(2) * prior).mean()

        return L


def train(cfg, model, dataloader, run_path, batch_to_device,
          optimizer):

    check_cp_exist = pjoin(run_path, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if(os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    test_im_dir = pjoin(run_path, 'recons')
    if(not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)

    # criterion = PriorMSELoss()
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(run_path)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                    cfg.lr_power)
    
    frames_tnsr_brd = np.linspace(0,
                                  len(dataloader) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)
    best_loss = float('inf')
    for epoch in range(cfg.epochs_autoenc):

            running_loss = 0.0

            prev_ims = []
            prev_ims_recons = []
            # Iterate over data.
            pbar = tqdm.tqdm(total=len(dataloader))
            for i, data in enumerate(dataloader):
                data = batch_to_device(data)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # im_recons, feats = model(data['image'])
                im_recons, _ = model(data['image'])

                # loss = criterion(im_recons, data['image'], data['prior'])
                loss = criterion(im_recons, data['image'])

                loss.backward()
                optimizer.step()

                prev_ims += [np.rollaxis(im.cpu().detach().numpy(), 0, 3)
                             for im in data['image']]
                prev_ims_recons += [np.rollaxis(im.cpu().detach().numpy(), 0, 3)
                                    for im in im_recons]

                running_loss += loss.cpu().detach().numpy()
                loss_ = running_loss / ((i + 1) * cfg.batch_size)
                pbar.set_description('epoch {}/{} lss: {:.6f} lr: {:.2f}'.format(
                    epoch+1,
                    cfg.epochs_autoenc,
                    loss_,
                    lr_sch.get_lr()[0]))

                pbar.update(1)

            pbar.close()
            writer.add_scalar('loss_autoenc',
                                loss_,
                                epoch)
            lr_sch.step()

            # save previews
            prev_ims = np.vstack([prev_ims[i] for i in frames_tnsr_brd])
            prev_ims_recons = np.vstack([prev_ims_recons[i] for i in frames_tnsr_brd])
            all = np.concatenate((prev_ims, prev_ims_recons), axis=1)

            io.imsave(
                pjoin(test_im_dir, 'im_{:04d}.png'.format(epoch)),
                all)
            # writer.add_image('autoenc',
            #                  all, epoch, dataformats='HWC')

            # save checkpoint
            is_best = False
            if (loss_ < best_loss):
                is_best = True
                best_loss = loss_
            path = pjoin(run_path, 'checkpoints')
            utls.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model,
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                },
                is_best,
                fname_cp='checkpoint_autoenc.pth.tar',
                fname_bm='best_autoenc.pth.tar',
                path=path)


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DeepLabv3Plus(pretrained=True, n_clusters=cfg.n_clusters)
    # model = DeepLabv3_plus(n_classes=3, pretrained=True)
    model.to(device)

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if(not os.path.exists(run_path)):
        os.makedirs(run_path)

    _, transf_normal = im_utils.make_data_aug(cfg)
    # transf_rescale = iaa.Sequential([
    #     rescale_augmenter])

    dl_train = Loader(pjoin(cfg.in_root, 'Dataset'+cfg.train_dir),
                      normalization=transf_normal)

    dataloader_train = DataLoader(dl_train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl_train.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)


    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    optimizer = optim.SGD(params=[
        {'params': model.parameters(), 'lr': cfg.lr_autoenc}
    ],
                          momentum=cfg.momentum,
                          weight_decay=cfg.decay)

    print('run_path: {}'.format(run_path))

    train(cfg, model, dataloader_train,
          run_path, batch_to_device, optimizer)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', requird=True)

    cfg = p.parse_args()

    main(cfg)
