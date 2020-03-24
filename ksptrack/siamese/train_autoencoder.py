from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
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
from ksptrack.models.unet_model import UNet
from ksptrack.siamese import im_utils
from skimage import io
import numpy as np
from ksptrack.models.my_augmenters import rescale_augmenter
from torch.nn.functional import sigmoid, tanh
from ksptrack.siamese.clustering import get_features
from ksptrack.utils.bagging import calc_bagging
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
import pandas as pd
from ksptrack.prev_trans_costs import colorize


def make_pm_prevs(model, dataloaders, cfg, centroids, all_labels, device):
    feats, labels_pos = get_features(model, dataloaders['all'], device)

    probas = calc_bagging(np.concatenate(feats),
                          np.concatenate(labels_pos),
                          T=cfg.bag_t,
                          bag_max_depth=cfg.bag_max_depth,
                          bag_n_feats=cfg.bag_n_feats)
    frames = [s['frame_idx'] for s in dataloaders['prev']]
    frames = [item for sublist in frames for item in sublist]

    df = centroids.assign(desc=np.concatenate(feats), proba=probas)
    scores = get_pm_array(all_labels, df, frames)
    scores_thr = [(s > 0.5).astype(float) for s in scores]
    scores = [colorize(s) for s in scores]
    scores_thr = [colorize(s) for s in scores_thr]
    images = [
        np.rollaxis(s['image'].squeeze().cpu().numpy(), 0, 3)
        for s in dataloaders['prev']
    ]
    all_images = (np.concatenate(images, axis=1) * 255).astype(np.uint8)
    all_scores = np.concatenate(scores, axis=1)
    all_scores_thr = np.concatenate(scores_thr, axis=1)
    all = np.concatenate((all_images, all_scores, all_scores_thr), axis=0)

    return all


def get_features(model, dataloader, device):
    # form initial cluster centres
    labels_pos_mask = []
    features = []

    roi_pool = SupPixPool()
    model.eval()
    model.to(device)
    print('getting features')
    pbar = tqdm.tqdm(total=len(dataloader))
    for index, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)
        with torch.no_grad():
            res = model(data['image'])

        pooled_aspp_feats = [
            roi_pool(res['aspp_feats'][b].unsqueeze(0),
                     data['labels'][b].unsqueeze(0)).squeeze().T
            for b in range(data['labels'].shape[0])
        ]

        pooled_aspp_feats = torch.cat(pooled_aspp_feats)
        features.append(pooled_aspp_feats.detach().cpu().numpy().squeeze())
        clicked_labels = [
            item for sublist in data['labels_clicked'] for item in sublist
        ]
        labels = data['labels'].cpu().numpy()
        to_add = np.zeros(np.unique(labels).shape[0]).astype(bool)
        to_add[clicked_labels] = True
        labels_pos_mask.append(to_add)

        pbar.update(1)
    pbar.close()

    return features, labels_pos_mask


def get_pm_array(labels, pm_df, frames=None):
    """ Returns array same size as labels with probabilities of bagging model
    """

    scores = labels.copy().astype(float)
    if (frames is None):  #Make all frames
        frames = np.arange(scores.shape[-1])
    else:
        frames = np.asarray(frames)

    i = 0
    bar = tqdm.tqdm(total=len(frames))
    for f in frames:
        this_frame_pm_df = pm_df[pm_df['frame'] == f]
        dict_keys = this_frame_pm_df['label']
        dict_vals = this_frame_pm_df['proba']
        dict_map = dict(zip(dict_keys, dict_vals))
        # Create 2D replacement matrix
        replace = np.array([list(dict_map.keys()), list(dict_map.values())])
        # Find elements that need replacement
        mask = np.isin(scores[..., f], replace[0, :])
        # Replace elements
        scores[mask, f] = replace[
            1, np.searchsorted(replace[0, :], scores[mask, f])]
        i += 1
        bar.update(1)
    bar.close()

    if (frames is None):
        return scores
    else:
        return [scores[..., f] for f in frames]


class PriorMSELoss(torch.nn.Module):
    def __init__(self):
        super(PriorMSELoss, self).__init__()

    def forward(self, y, y_true, prior):

        L = ((y - y_true).pow(2) * prior).mean()

        return L


def train(cfg, model, dataloaders, run_path, device, optimizer):

    all_labels = np.array(
        [s['labels'].cpu().squeeze().numpy() for s in dataloaders['all']])
    all_labels = np.rollaxis(all_labels, 0, 3)
    centroids = pd.read_pickle(
        pjoin(cfg.in_root, 'Dataset' + cfg.train_dir, 'precomp_desc',
              'sp_desc_autoenc.p'))

    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    check_cp_exist = pjoin(run_path, 'checkpoints',
                           'checkpoint_autoenc.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    test_im_dir = pjoin(run_path, 'recons')
    pm_im_dir = pjoin(run_path, 'recons_pm')
    if (not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)
    if (not os.path.exists(pm_im_dir)):
        os.makedirs(pm_im_dir)

    criterion = torch.nn.MSELoss()
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
                    '[{}] epch {}/{} lss: {:.6f} lr: {:.5f}'.format(
                        phase, epoch + 1, cfg.epochs_autoenc,
                        loss_ if phase == 'train' else 0,
                        lr_sch.get_lr()[0] if phase == 'train' else 0))

                pbar.update(1)
            if (phase == 'train'):
                writer.add_scalar('loss_autoenc', loss_, epoch)

                if ((epoch + 1) % cfg.cp_period == 0):
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

                all = make_pm_prevs(model, dataloaders, cfg, centroids, all_labels,
                                    device)

                io.imsave(pjoin(pm_im_dir, 'ep_{:04d}.png'.format(epoch)), all)


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DeepLabv3Plus(pretrained=False)
    model.to(device)

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    transf, transf_normal = im_utils.make_data_aug(cfg)

    dl = LocPriorDataset(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                         augmentations=transf,
                         normalization=transf_normal)

    cfg.batch_size = 2
    dataloader_train = DataLoader(dl,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)

    dl_all_prev = LocPriorDataset(pjoin(cfg.in_root,
                                        'Dataset' + cfg.train_dir),
                                  normalization=transf_normal)
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

    # optimizer = optim.SGD(
    #     params=[
    #         {'params': model.parameters(), 'lr': cfg.lr_autoenc},
    #     ],
    #     momentum=cfg.momentum,
    #     weight_decay=cfg.decay,
    #     )
    optimizer = optim.Adam(
        params=[
            {
                'params': model.parameters(),
                'lr': cfg.lr_autoenc
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
