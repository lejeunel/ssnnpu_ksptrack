from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import ksptrack.pu.utils as utls
import tqdm
from ksptrack.pu.modeling.unet import UNet
from ksptrack.pu import im_utils
from skimage import io
import numpy as np
from torch.nn.functional import sigmoid
from ksptrack.utils.bagging import calc_bagging
from ksptrack.utils.my_utils import get_pm_array
import pandas as pd
from ksptrack.pu.im_utils import colorize, get_features


def make_pm_prevs(model, dataloaders, cfg, centroids, all_labels, device):
    res = get_features(model, dataloaders['all'], device)
    feats = res['feats']
    labels_pos = res['labels_pos_mask']

    probas = calc_bagging(np.concatenate(feats),
                          np.concatenate(labels_pos),
                          T=cfg.bag_t,
                          bag_max_depth=cfg.bag_max_depth,
                          bag_n_feats=cfg.bag_n_feats)
    frames = [s['frame_idx'] for s in dataloaders['prev']]
    frames = [item for sublist in frames for item in sublist]

    df = centroids.assign(desc=np.concatenate(feats), proba=probas)
    scores = get_pm_array(all_labels, df)
    scores = [scores[f] for f in frames]
    scores_thr = [(s > 0.5).astype(float) for s in scores]
    scores = [colorize(s) for s in scores]
    scores_thr = [colorize(s) for s in scores_thr]
    images = [
        np.rollaxis(
            (255 * s['image'].squeeze().cpu().numpy()).astype(np.uint8), 0, 3)
        for s in dataloaders['prev']
    ]
    all_images = (np.concatenate(images, axis=1)).astype(np.uint8)
    all_scores = np.concatenate(scores, axis=1)
    all_scores_thr = np.concatenate(scores_thr, axis=1)
    all = np.concatenate((all_images, all_scores, all_scores_thr), axis=0)

    return all


class PriorMSELoss(torch.nn.Module):
    def __init__(self):
        super(PriorMSELoss, self).__init__()

    def forward(self, y, y_true, prior):

        L = ((y - y_true).pow(2) * prior).mean()

        return L


def train(cfg, model, dataloaders, run_path, device, optimizer):

    all_labels = np.array(
        [s['labels'].cpu().squeeze().numpy() for s in dataloaders['all']])
    centroids = pd.read_pickle(
        pjoin(cfg.in_root, 'Dataset' + cfg.train_dir, 'precomp_desc',
              'sp_desc_autoenc.p'))

    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    check_cp_exist = pjoin(run_path, 'checkpoints', 'cp_autoenc.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    test_im_dir = pjoin(run_path, 'recons')
    pm_im_dir = pjoin(run_path, 'recons_pm')
    if (not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)
    if (not os.path.exists(pm_im_dir)):
        os.makedirs(pm_im_dir)

    # cfg.prev_period = 2

    criterion = torch.nn.MSELoss()
    # criterion = PriorMSELoss()
    writer = SummaryWriter(run_path)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.lr_power)
    best_loss = float('inf')
    for epoch in range(cfg.epochs_autoenc):
        for phase in ['train', 'prev']:

            running_loss = 0.0

            prev_ims = {}
            prev_ims_recons = {}
            # Iterate over data.
            pbar = tqdm.tqdm(total=len(dataloaders[phase]))
            for i, data in enumerate(dataloaders[phase]):
                if (phase == 'train'):
                    model.train()
                elif (phase == 'prev'):
                    model.eval()
                data = batch_to_device(data)

                with torch.set_grad_enabled(phase == 'train'):
                    res = model(data['image'])

                if (phase == 'train'):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    loss = criterion(sigmoid(res['output']), data['image'])

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
                        np.rollaxis(
                            sigmoid(res['output'])[0].cpu().detach().numpy(),
                            0, 3)
                    })

                pbar.set_description(
                    '[{}] ep {}/{} lss: {:.3e} lr: {:.1e}'.format(
                        phase, epoch + 1, cfg.epochs_autoenc,
                        loss_ if phase == 'train' else 0,
                        lr_sch.get_lr()[0] if phase == 'train' else 0))

                pbar.update(1)
            if (phase == 'train'):
                writer.add_scalar('loss_autoenc', loss_, epoch)
                # save checkpoint
                is_best = False
                if (loss_ < best_loss):
                    is_best = True
                    best_loss = loss_
                path = pjoin(run_path, 'checkpoints', 'cp_autoenc.pth.tar')
                utls.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()
                    }, path)

            pbar.close()
            lr_sch.step()

            if ((phase == 'prev') & ((epoch + 1) % cfg.prev_period == 0)):

                # save previews
                prev_ims = np.vstack(
                    [prev_ims[k] for k in sorted(prev_ims.keys())])
                prev_ims_recons = np.vstack([
                    prev_ims_recons[k] for k in sorted(prev_ims_recons.keys())
                ])
                all = np.concatenate((prev_ims, prev_ims_recons), axis=1)

                io.imsave(pjoin(test_im_dir, 'ep_{:04d}.png'.format(epoch)),
                          all)

                all = make_pm_prevs(model, dataloaders, cfg, centroids,
                                    all_labels, device)

                io.imsave(pjoin(pm_im_dir, 'ep_{:04d}.png'.format(epoch)), all)


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = UNet(depth=5, skip_mode='none', use_coordconv=True)
    model.to(device)

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    transf, _ = im_utils.make_data_aug(cfg)

    dl = LocPriorDataset(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                         augmentations=transf,
                         normalization='rescale',
                         resize_shape=cfg.in_shape)

    cfg.batch_size = 2
    dataloader_train = DataLoader(dl,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl.collate_fn,
                                  drop_last=True)

    dl_all_prev = LocPriorDataset(pjoin(cfg.in_root,
                                        'Dataset' + cfg.train_dir),
                                  normalization='rescale',
                                  resize_shape=cfg.in_shape)

    dataloader_all_prev = DataLoader(dl_all_prev, collate_fn=dl.collate_fn)
    dl_prev = Subset(
        dl_all_prev, np.linspace(0, len(dl) - 1, num=cfg.n_ims_test,
                                 dtype=int))
    dataloader_prev = DataLoader(dl_prev, collate_fn=dl.collate_fn)

    dataloaders = {
        'train': dataloader_train,
        'all': dataloader_all_prev,
        'prev': dataloader_prev
    }

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    optimizer = optim.Adam(
        params=[
            {
                'params': model.parameters(),
                'lr': 1e-4
            },
        ],
        weight_decay=cfg.decay,
    )
    print('run_path: {}'.format(run_path))

    train(cfg, model, dataloaders, run_path, device, optimizer)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
