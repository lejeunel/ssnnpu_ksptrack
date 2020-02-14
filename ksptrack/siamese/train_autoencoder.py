from loader import Loader
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
import tqdm
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.siamese import im_utils
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


class PriorMSELoss(torch.nn.Module):
    def __init__(self):
        super(PriorMSELoss, self).__init__()

    def forward(self, y, y_true, prior):

        L = ((y - y_true).pow(2) * prior).mean()

        return L


def train(cfg, model, dataloaders, run_path, batch_to_device, optimizer):

    check_cp_exist = pjoin(run_path, 'checkpoints',
                           'checkpoint_autoenc.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    test_im_dir = pjoin(run_path, 'recons')
    if (not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)

    # criterion = PriorMSELoss()
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(run_path)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.lr_power)

    best_loss = float('inf')
    for epoch in range(cfg.epochs_autoenc):
        for phase in dataloaders.keys():

            running_loss = 0.0

            prev_ims = {}
            prev_ims_recons = {}
            # Iterate over data.
            pbar = tqdm.tqdm(total=len(dataloaders[phase]))
            for i, data in enumerate(dataloaders[phase]):
                if (phase == 'train'):
                    model.train()
                else:
                    model.eval()
                data = batch_to_device(data)

                with torch.set_grad_enabled(phase == 'train'):
                    res = model(data['image'])

                if (phase == 'train'):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # loss = criterion(res['output'], data['image'], data['prior'])
                    loss = criterion(res['output'], data['image'])

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.cpu().detach().numpy()
                    loss_ = running_loss / ((i + 1) * cfg.batch_size)

                else:
                    prev_ims.update({
                        data['frame_name'][0]:
                        np.rollaxis(data['image'][0].cpu().detach().numpy(), 0,
                                    3)
                    })
                    prev_ims_recons.update({
                        data['frame_name'][0]:
                        np.rollaxis(res['output'][0].cpu().detach().numpy(), 0, 3)
                    })

                pbar.set_description(
                    '[{}] epch {}/{} lss: {:.6f} lr: {:.2f}'.format(
                        phase, epoch + 1, cfg.epochs_autoenc,
                        loss_ if phase == 'train' else 0,
                        lr_sch.get_lr()[0] if phase == 'train' else 0))

                pbar.update(1)

            pbar.close()
            if (phase == 'train'):
                writer.add_scalar('loss_autoenc', loss_, epoch)
                lr_sch.step()

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
            else:

                # save previews
                prev_ims = np.vstack([prev_ims[k] for k in sorted(prev_ims.keys())])
                prev_ims_recons = np.vstack(
                    [prev_ims_recons[k] for k in sorted(prev_ims_recons.keys())])
                all = np.concatenate((prev_ims, prev_ims_recons), axis=1)

                io.imsave(pjoin(test_im_dir, 'im_{:04d}.png'.format(epoch)),
                          all)


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DeepLabv3Plus(pretrained=False)
    model.to(device)

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    _, transf_normal = im_utils.make_data_aug(cfg)
    # transf_rescale = iaa.Sequential([
    #     rescale_augmenter])

    dl = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                normalization=transf_normal)

    prev_sampler = SubsetRandomSampler(
        np.random.choice(len(dl), size=cfg.n_ims_test, replace=False))
    dataloader_prev = DataLoader(dl,
                                 sampler=prev_sampler,
                                 collate_fn=dl.collate_fn)

    dataloader_train = DataLoader(dl,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)
    dataloaders = {'train': dataloader_train, 'prev': dataloader_prev}

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    optimizer = optim.SGD(params=[{
        'params': model.parameters(),
        'lr': cfg.lr_autoenc
    }],
                          momentum=cfg.momentum,
                          weight_decay=cfg.decay)

    print('run_path: {}'.format(run_path))

    train(cfg, model, dataloaders, run_path, batch_to_device, optimizer)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', requird=True)

    cfg = p.parse_args()

    main(cfg)
