import os
from glob import glob
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import yaml
from imgaug import augmenters as iaa
from ksptrack.modeling.unet import UNet, init_weights_normal
from ksptrack.pu import utils as utls
from ksptrack.pu.im_utils import get_features, sp_pool
from ksptrack.pu.losses import BalancedBCELoss, PULoss
from ksptrack.pu.plots import freq_vs_epc
from ksptrack.pu.pu_utils import init_kfs, update_priors_kf
from ksptrack.utils import prev_trans_costs
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from skimage import io
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from ksptrack import params

torch.backends.cudnn.benchmark = True


def init_last_layer(model, p, device):
    offset = -np.log((1 - p) / p)
    print('setting objectness prior to {}, p={}'.format(offset, p))
    model.decoder.output_convolution[0].bias.data.fill_(
        torch.tensor(offset).to(device))

    return model


def do_previews(dataloaders, writer, out_paths, cfg, epoch):

    cfg.do_all = False
    cfg.fin = dataloaders['prev']
    res_prevs = prev_trans_costs.main(cfg)
    prev_ims = np.concatenate([
        np.concatenate((s['image'], s['pm'], s['pm_thr']), axis=1)
        for s in res_prevs['images']
    ],
                              axis=0)
    io.imsave(pjoin(out_paths['prevs'], 'ep_{:04d}.png'.format(epoch)),
              prev_ims)

    if cfg.do_scores:
        writer.add_scalar('F1/{}'.format(cfg.exp_name),
                          res_prevs['scores']['f1'], epoch)
        writer.add_scalar('auc/{}'.format(cfg.exp_name),
                          res_prevs['scores']['auc'], epoch)

    io.imsave(pjoin(out_paths['prevs'], 'ep_{:04d}.png'.format(epoch)),
              prev_ims)


def make_data_aug(cfg):
    transf = iaa.Sequential([
        iaa.OneOf([
            iaa.BilateralBlur(d=8,
                              sigma_color=(cfg.aug_blur_color_low,
                                           cfg.aug_blur_color_high),
                              sigma_space=(cfg.aug_blur_space_low,
                                           cfg.aug_blur_space_high)),
            iaa.AdditiveGaussianNoise(scale=(0, cfg.aug_noise * 255)),
            iaa.GammaContrast((cfg.aug_gamma_low, cfg.aug_gamma_high))
        ])
        # iaa.Flipud(p=0.5),
        # iaa.Fliplr(p=.5),
        # iaa.Rot90((1, 3))
    ])
    if cfg.phase == 2:
        return iaa.Noop()

    return transf


def make_loss(cfg):

    if cfg.loss_obj_pred == 'pu':

        criterion = PULoss(do_ascent=cfg.nnpu_ascent, pxls=cfg.pxls)

    elif cfg.loss_obj_pred == 'bce':

        criterion = BalancedBCELoss()
        # loss = criterion(input, target, pi=pi, pi_mul=cfg.pi_mul)

    return criterion


def train_one_epoch(model, dataloader, optimizer, device, epoch, lr_sch, cfg,
                    priors, writer):

    model.train()

    running_loss = 0.0
    running_loss_pu = 0.0
    running_neg_risk = 0.0
    running_pos_risk = 0.0
    running_means = 0.0

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            res = model(data['image'])

            criterion = make_loss(cfg)
            frames = data['frame_idx']
            pi = [priors[-1][f] for f in frames]
            target = [
                data['annotations'][data['annotations']['frame'] == f]
                for f in frames
            ]
            if not cfg.pxls:
                n_labels = [
                    data['annotations'][data['annotations']['frame'] ==
                                        f].iloc[0]['n_labels'] for f in frames
                ]
                inp = sp_pool(res['output'], data['labels'])
                inp = torch.split(inp.squeeze(), n_labels)
            else:
                inp = res['output']

            loss_pu = criterion(inp, target, pi=pi)
            loss = loss_pu['loss']
            mean = torch.cat(inp).sigmoid().mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.clip_grad_norm)

            optimizer.step()

        niter = epoch * len(dataloader) + i

        running_loss += loss.cpu().detach().numpy().item()
        running_loss_pu += loss_pu['loss'].cpu().detach().numpy().item()
        running_pos_risk += loss_pu['pos_risk'].cpu().detach().numpy().item()
        running_neg_risk += loss_pu['neg_risk'].cpu().detach().numpy().item()
        running_means += mean.cpu().detach().numpy().item()

        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        loss_pu = running_loss_pu / ((i + 1) * cfg.batch_size)
        pos_risk = running_pos_risk / ((i + 1) * cfg.batch_size)
        neg_risk = running_neg_risk / ((i + 1) * cfg.batch_size)
        means = running_means / (i + 1)

        writer.add_scalar('train/loss', loss_, niter)
        writer.add_scalar('train/loss_pu', loss_pu, niter)
        writer.add_scalar('train/pos_risk', pos_risk, niter)
        writer.add_scalar('train/neg_risk', neg_risk, niter)
        writer.add_scalar('train/means', means, niter)
        writer.add_scalar('lr/{}'.format(cfg.exp_name),
                          lr_sch.get_last_lr()[0], epoch)

        pbar.set_description('lss {:.6e}'.format(loss_))
        pbar.update(1)

    pbar.close()
    lr_sch.step()


def train(cfg, model, device, dataloaders, run_path):

    cp_fname = 'cp.pth.tar'
    out_paths = {
        'prevs': pjoin(run_path, 'prevs'),
        'prevs_data': pjoin(run_path, 'prevs_data'),
        'curves': pjoin(run_path, 'curves'),
        'curves_data': pjoin(run_path, 'curves_data'),
        'cps': pjoin(run_path, 'cps')
    }
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    for p in out_paths.values():
        if not os.path.exists(p):
            os.makedirs(p)

    check_cp_exist = pjoin(out_paths['cps'], 'cp_{:04d}.pth.tar'.format(0))
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    if cfg.true_prior:
        print('using groundtruth for class-priors')
        res = get_features(model, dataloaders['init'], device)
        truths = res['truths_unpooled'] if cfg.pxls else res['truths']
        priors = np.array([(t > 0).sum() / t.size for t in truths])

        if cfg.true_prior == 'max':
            print('using only max of groundtruth')
            training_priors = [
                np.ones_like(priors) * priors.max() * cfg.pi_post_ratio_truth *
                cfg.pi_overspec_ratio
            ]
        elif cfg.true_prior == 'mean':
            print('using mean of groundtruth')
            training_priors = [
                np.ones_like(priors) * priors.mean() *
                cfg.pi_post_ratio_truth * cfg.pi_overspec_ratio
            ]
        else:
            print('using groundtruth per-frame')
            training_priors = [cfg.pi_post_ratio_truth * priors]

        print('reinitializing weights')
        model.apply(init_weights_normal)
        model = init_last_layer(model, 0.01, device)

    elif cfg.phase == 2:
        print('using model of alternate phase')

        check_cp_exist = pjoin(out_paths['cps'],
                               'cp_{:04d}.pth.tar'.format(cfg.cp_period))
        if (os.path.exists(check_cp_exist)):
            print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
            return
        from argparse import Namespace
        cfg_prior = Namespace(root_path=cfg.out_path,
                              train_dirs=['Dataset' + cfg.train_dir],
                              exp_name=cfg.pred_init_dir,
                              thr=cfg.var_thr,
                              rho_pi_err=cfg.rho_pi_err,
                              n_epc=cfg.var_epc,
                              min_epc=cfg.min_var_epc,
                              save=pjoin(run_path, 'priors.png'),
                              title='',
                              curves_dir='curves_data')
        priors, ep = freq_vs_epc.main(cfg_prior)
        training_priors = [priors[0] * cfg.pi_post_ratio]
        ep = ep[0]
        print('taking priors from epoch {}'.format(ep))
        print('pi_post_ratio {}'.format(cfg.pi_post_ratio))

        cps = sorted(
            glob(
                pjoin(cfg.out_path, 'Dataset' + cfg.train_dir,
                      cfg.pred_init_dir, 'cps', '*.pth.tar')))
        cp_eps = np.array([
            int(os.path.split(f)[-1].split('_')[-1].split('.')[0]) for f in cps
        ])
        cp_fname = cps[np.argmin(np.abs(cp_eps - ep))]

        path_ = pjoin(os.path.split(run_path)[0], cfg.pred_init_dir, cp_fname)
        print('loading checkpoint {}'.format(path_))
        state_dict = torch.load(path_,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    elif cfg.phase == 1:
        check_cp_exist = pjoin(out_paths['cps'],
                               'cp_{:04d}.pth.tar'.format(cfg.cp_period))
        if (os.path.exists(check_cp_exist)):
            print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
            return
        print('using model of warm-up phase')
        res = get_features(model, dataloaders['init'], device)
        truths = res['truths_unpooled'] if cfg.pxls else res['truths']
        max_freq = cfg.pi_overspec_ratio * np.max([(truth.sum()) / truth.size
                                                   for truth in truths])
        cfg.init_pi = float(max_freq)
        print('using constant priors: {}'.format(cfg.init_pi))
        filter, state_means, state_covs = init_kfs(n_frames=len(truths),
                                                   init_mean=max_freq,
                                                   init_cov=0.03,
                                                   cfg=cfg)
        training_priors = state_means
        assert cfg.pred_init_dir, 'give pred_init_dir'

        cp = sorted(
            glob(pjoin(cfg.out_path, cfg.pred_init_dir, 'cps',
                       '*.pth.tar')))[-1]

        print('loading checkpoint {}'.format(cp))
        state_dict = torch.load(cp, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    else:
        check_cp_exist = pjoin(out_paths['cps'],
                               'cp_{:04d}.pth.tar'.format(cfg.epochs_pre_pred))
        if (os.path.exists(check_cp_exist)):
            print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
            return

        print('doing warm-up phase')
        # print('reinitializing weights')
        model.apply(init_weights_normal)

        res = get_features(model, dataloaders['init'], device)
        truths = res['truths_unpooled'] if cfg.pxls else res['truths']
        max_freq = cfg.pi_overspec_ratio * np.max([(truth.sum()) / truth.size
                                                   for truth in truths])

        model = init_last_layer(model, 0.01, device)

        cfg.init_pi = float(max_freq)
        init_prior = cfg.init_pi / 2
        print('pi_max: {}'.format(cfg.init_pi))
        print('using initial training priors: {}'.format(init_prior))

        training_priors = [
            np.ones(len(dataloaders['init'].dataset)) * init_prior
        ]

    path = pjoin(out_paths['cps'], 'cp_{:04d}.pth.tar'.format(0))

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    writer = SummaryWriter(run_path, flush_secs=1)

    best_loss = float('inf')

    model.to(device)

    gamma_lr = 1.
    if cfg.phase == 0:
        lr = cfg.lr0
        n_epochs = cfg.epochs_pre_pred
    elif cfg.phase == 1:
        lr = cfg.lr1
        n_epochs = cfg.epochs_pred
    else:
        lr = cfg.lr2_start
        n_epochs = cfg.epochs_post_pred

        if cfg.true_prior:
            gamma_lr = cfg.lr_gamma
            lr = lr * 10

    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=cfg.decay,
                           eps=cfg.eps)

    lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[cfg.lr_epoch_milestone_0], gamma=gamma_lr)

    for epoch in range(n_epochs):

        print('epoch {}/{}, mode: {}, lr: {:.2e}'.format(
            epoch, n_epochs, 'pred',
            lr_sch.get_last_lr()[0]))

        # save checkpoint
        if (epoch > 0) and (epoch % cfg.cp_period == 0):
            path_cp = pjoin(out_paths['cps'],
                            'cp_{:04d}.pth.tar'.format(epoch))
            utls.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model,
                    'best_loss': best_loss,
                }, path_cp)
        if (epoch % cfg.cp_period == 0) and (epoch > 0):
            cfg.model_path = path_cp
            do_previews(dataloaders, writer, out_paths, cfg, epoch)

        if (cfg.phase == 1) and (epoch > 0) and (epoch % cfg.prior_period
                                                 == 0):
            filter, state_means, state_covs = update_priors_kf(
                model, dataloaders['init'], device, state_means, state_covs,
                filter, writer, out_paths['curves'], out_paths['curves_data'],
                epoch, cfg)
            training_priors = state_means

        train_one_epoch(model,
                        dataloaders['train'],
                        optimizer,
                        device,
                        epoch,
                        lr_sch,
                        cfg,
                        priors=training_priors,
                        writer=writer)

    # save previews
    do_previews(dataloaders, writer, out_paths, cfg, epoch + 1)

    utls.save_checkpoint({
        'epoch': epoch + 1,
        'model': model,
        'best_loss': 0,
    }, pjoin(out_paths['cps'], 'cp_{:04d}.pth.tar'.format(epoch + 1)))

    return model


def main(cfg):

    run_path = pjoin(cfg.out_path, cfg.exp_name)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = UNet(out_channels=1, in_channels=3).to(device)

    transf = make_data_aug(cfg)

    dl_train = LocPriorDataset(cfg.in_path,
                               normalization='rescale',
                               augmentations=transf,
                               locs_dir=cfg.locs_dir,
                               locs_fname=cfg.locs_fname,
                               resize_shape=cfg.in_shape)
    dl_init = LocPriorDataset(cfg.in_path,
                              normalization='rescale',
                              locs_dir=cfg.locs_dir,
                              locs_fname=cfg.locs_fname,
                              resize_shape=cfg.in_shape)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_train) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_train = DataLoader(dl_train,
                                  collate_fn=dl_train.collate_fn,
                                  shuffle=True,
                                  drop_last=True,
                                  batch_size=cfg.batch_size)
    dataloader_init = DataLoader(dl_init, collate_fn=dl_init.collate_fn)

    dataloaders = {
        'train': dataloader_train,
        'prev': frames_tnsr_brd,
        'init': dataloader_init
    }

    model = train(cfg, model, device, dataloaders, run_path)

    return model


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
