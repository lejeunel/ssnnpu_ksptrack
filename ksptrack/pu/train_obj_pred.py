import os
from glob import glob
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import yaml
from imgaug import augmenters as iaa
from skimage import io
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from ksptrack import prev_trans_costs
from ksptrack.cfgs import params as params_ksp
from ksptrack.pu import utils as utls
from ksptrack.pu.im_utils import get_features, sp_pool
from ksptrack.pu.loader import Loader
from ksptrack.pu.losses import BalancedBCELoss, PULoss
from ksptrack.pu.modeling.unet import UNet, init_weights_normal
from ksptrack.pu.plots import freq_vs_epc
from ksptrack.pu.set_explorer import SetExplorer
from ksptrack.pu.tree_set_explorer import TreeSetExplorer
from pykalman import KalmanFilter

torch.backends.cudnn.benchmark = True


def do_previews(model, dataloaders, device, writer, out_paths, cfg, cfg_ksp,
                epoch, priors):

    cfg_ksp.use_model_pred = True
    if cfg.sec_phase == False and epoch == 0:
        cfg_ksp.use_model_pred = False

    print('generating previews. Using model pred: {}'.format(
        cfg_ksp.use_model_pred))

    cfg_ksp.aug_df_path = pjoin(out_paths['aug_sets'],
                                'aug_df_ep_{:04d}.p'.format(epoch))
    dataloaders['train'].dataset.positives.to_pickle(cfg_ksp.aug_df_path)
    res_prevs = prev_trans_costs.main(cfg_ksp)
    prev_ims = np.concatenate([
        np.concatenate((s['image'], s['pm'], s['pm_thr']), axis=1)
        for s in res_prevs['images']
    ],
                              axis=0)
    io.imsave(pjoin(out_paths['prevs'], 'ep_{:04d}.png'.format(epoch)),
              prev_ims)
    writer.add_scalar('F1/{}'.format(cfg.exp_name), res_prevs['scores']['f1'],
                      epoch)
    writer.add_scalar('auc/{}'.format(cfg.exp_name),
                      res_prevs['scores']['auc'], epoch)

    io.imsave(pjoin(out_paths['prevs'], 'ep_{:04d}.png'.format(epoch)),
              prev_ims)


def augment(model,
            dataloaders,
            device,
            writer,
            na,
            epoch,
            cfg,
            priors,
            do_reset=False):

    print('augmenting positive set')
    res = get_features(model, dataloaders['init'], device)
    losses = [-np.log(1 - o + 1e-8) for o in res['outs']]
    dataloaders['train'].dataset.make_candidates(losses)

    if do_reset:
        print('resetting augmented set')
        dataloaders['train'].dataset.reset_augs()
        print('augmenting set by {} samples'.format(na))
        dataloaders['train'].dataset.augment_positives(na, priors,
                                                       cfg.unlabeled_ratio)

    else:
        curr_augs = dataloaders['train'].dataset.n_aug
        print('incrementing augmented set by {} samples'.format(na -
                                                                curr_augs))
        dataloaders['train'].dataset.augment_positives(na - curr_augs, priors,
                                                       cfg.unlabeled_ratio)

    print(dataloaders['train'].dataset)
    writer.add_scalar('n_aug/{}'.format(cfg.exp_name),
                      dataloaders['train'].dataset.n_aug, epoch)
    writer.add_scalar('n_pos/{}'.format(cfg.exp_name),
                      dataloaders['train'].dataset.n_pos, epoch)
    writer.add_scalar('aug_purity/{}'.format(cfg.exp_name),
                      dataloaders['train'].dataset.ratio_purity_augs, epoch)


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

    return transf


def make_loss(cfg):

    if cfg.loss_obj_pred == 'pu':

        if (cfg.unlabeled_ratio > 0) and cfg.pxls:
            raise ValueError('when augmenting, set pxls to False')

        criterion = PULoss(do_ascent=cfg.nnpu_ascent,
                           aug_in_neg=cfg.aug_in_neg,
                           pxls=cfg.pxls)

    elif cfg.loss_obj_pred == 'bce':

        criterion = BalancedBCELoss()
        # loss = criterion(input, target, pi=pi, pi_mul=cfg.pi_mul)

    return criterion


def train_one_epoch(model, dataloader, optimizer, device, epoch, lr_sch, cfg,
                    priors, writer):

    model.train()

    running_loss = 0.0

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

            loss = criterion(inp, target, pi=pi)

            loss.backward()

            optimizer.step()

        running_loss += loss.cpu().detach().numpy()
        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        niter = epoch * len(dataloader) + i
        writer.add_scalar('train/loss_{}'.format(cfg.exp_name), loss_, niter)
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
        'aug_sets': pjoin(run_path, 'aug_sets'),
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

    print('computing priors')
    if cfg.true_prior:
        print('using groundtruth')
        res = get_features(model,
                           dataloaders['init'],
                           device,
                           loc_prior=cfg.loc_prior)
        truths = res['truths_unpooled'] if cfg.pxls else res['truths']
        priors = np.array([(t > 0).sum() / t.size for t in truths])
        priors = [priors]
        cps = sorted(
            glob(
                pjoin(cfg.out_root, 'Dataset' + cfg.train_dir,
                      cfg.pred_init_dir, 'cps', '*.pth.tar')))

        path_ = pjoin(os.path.split(run_path)[0], cfg.pred_init_dir, cps[-1])
        print('loading checkpoint {}'.format(path_))
        state_dict = torch.load(path_,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    elif cfg.sec_phase:
        print('using model of alternate phase')
        assert cfg.pred_init_dir, 'give pred_init_dir'
        from argparse import Namespace
        cfg_prior = Namespace(root_path=cfg.out_root,
                              train_dirs=['Dataset' + cfg.train_dir],
                              exp_name=cfg.pred_init_dir,
                              thr=cfg.var_thr,
                              n_epc=cfg.var_epc,
                              min_epc=cfg.min_var_epc,
                              save=pjoin(run_path, 'priors.png'),
                              title='',
                              curves_dir='curves_data')
        priors, ep = freq_vs_epc.main(cfg_prior)
        priors = [priors[0]]
        ep = ep[0]
        print('taking priors from epoch {}'.format(ep))
        cps = sorted(
            glob(
                pjoin(cfg.out_root, 'Dataset' + cfg.train_dir,
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

    else:
        res = get_features(model,
                           dataloaders['init'],
                           device,
                           loc_prior=cfg.loc_prior)
        truths = res['truths_unpooled'] if cfg.pxls else res['truths']
        max_freq = cfg.pi_overspec_ratio * np.max([(truth.sum()) / truth.size
                                                   for truth in truths])
        cfg.init_pi = float(max_freq)
        cfg.pi_xi = cfg.init_pi / 50
        cfg.pi_min = cfg.pi_xi
        print('using constant priors: {}'.format(cfg.init_pi))
        print('pi_xi: {}'.format(cfg.pi_xi))
        print('gradients clipped to norms: {}'.format(cfg.pi_min))
        pi_0 = np.array([cfg.init_pi for _ in res['labels_pos_mask']])
        priors = [pi_0]
        model_priors = [pi_0]

    n_pos = dataloaders['train'].dataset.n_pos
    if (cfg.unlabeled_ratio > 0):
        res = get_features(model,
                           dataloaders['init'],
                           device,
                           loc_prior=cfg.loc_prior)
        max_n_augs = cfg.unlabeled_ratio * (
            np.mean(priors[-1]) *
            np.concatenate(res['labels_pos_mask']).shape[0]) - n_pos
        max_n_augs = int(max(max_n_augs, 0))
    else:
        max_n_augs = 0

    first_aug_step = np.round(
        max_n_augs / cfg.epochs_pred).astype(int) * cfg.unlabeled_period
    n_augs = np.linspace(first_aug_step, max_n_augs,
                         cfg.epochs_pred).astype(int)
    print('will add {} augmented samples.'.format(max_n_augs))

    path = pjoin(out_paths['cps'], 'cp_{:04d}.pth.tar'.format(0))
    utls.save_checkpoint({
        'epoch': 0,
        'model': model,
        'best_loss': 0,
    }, path)

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    p_ksp = params_ksp.get_params('../cfgs')
    p_ksp.add('--model-path', default='')
    p_ksp.add('--in-path', default='')
    p_ksp.add('--do-all', default=False)
    p_ksp.add('--return-dict', default=False)
    p_ksp.add('--fin', nargs='+')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]
    cfg_ksp.bag_t = 600
    cfg_ksp.pm_thr = 0.5
    cfg_ksp.bag_n_feats = cfg.bag_n_feats
    cfg_ksp.bag_max_depth = cfg.bag_max_depth
    cfg_ksp.model_path = pjoin(run_path, cp_fname)
    cfg_ksp.trans_path = pjoin(os.path.split(run_path)[0], 'autoenc', cp_fname)
    cfg_ksp.use_model_pred = True if cfg.sec_phase else False
    cfg_ksp.trans = 'lfda'
    cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
    cfg_ksp.precomp_desc_path = pjoin(cfg_ksp.in_path, 'precomp_desc')
    cfg_ksp.fin = dataloaders['prev']
    cfg_ksp.sp_labels_fname = 'sp_labels.npy'
    cfg_ksp.do_scores = True
    cfg_ksp.loc_prior = cfg.loc_prior
    cfg_ksp.coordconv = cfg.coordconv
    cfg_ksp.n_augs = 0
    cfg_ksp.aug_method = cfg.aug_method

    writer = SummaryWriter(run_path, flush_secs=1)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_pred))

    if cfg.pred_init_dir:
        pass
    else:
        print('doing first phase')
        print('reinitializing weights')
        model.apply(init_weights_normal)

        offset = -np.log((1 - cfg.init_pi) / cfg.init_pi)
        print('setting objectness prior to {}, p={}'.format(
            offset, cfg.init_pi / 2))
        model.decoder.output_convolution[0].bias.data.fill_(
            torch.tensor(offset).to(device))

    model.to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.lr2 if cfg.sec_phase else cfg.lr1,
                          weight_decay=1e-2)

    lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[cfg.lr_epoch_milestone_0], gamma=cfg.lr_gamma)

    n_epochs = cfg.epochs_pred

    model_priors = []

    for na, epoch in zip(n_augs, range(n_epochs)):
        if na > 0:
            if (epoch == 0) or (epoch % cfg.unlabeled_period == 0):
                augment(model,
                        dataloaders,
                        device,
                        writer,
                        na,
                        epoch,
                        cfg,
                        priors[-1],
                        do_reset=cfg.aug_reset)

        n_pos = dataloaders['train'].dataset.n_pos
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
            cfg_ksp.model_path = path_cp
        if (epoch % (2 * cfg.cp_period) == 0) and (epoch > 0):
            do_previews(model, dataloaders, device, writer, out_paths, cfg,
                        cfg_ksp, epoch, priors[-1])

        if (not cfg.sec_phase) and (epoch > 0) and (not cfg.true_prior) and (
                epoch %
                cfg.prior_period == 0) and (epoch > cfg.epochs_pre_pred):
            priors, model_priors = update_priors(model,
                                                 dataloaders['init'],
                                                 device,
                                                 priors,
                                                 model_priors,
                                                 writer,
                                                 out_paths['curves'],
                                                 out_paths['curves_data'],
                                                 epoch,
                                                 cfg,
                                                 grad_method='clip',
                                                 inval_mode='copydecrease',
                                                 decrease_pimax=False)

        train_one_epoch(model,
                        dataloaders['train'],
                        optimizer,
                        device,
                        epoch,
                        lr_sch,
                        cfg,
                        priors=priors,
                        writer=writer)

    # save previews
    do_previews(model, dataloaders, device, writer, out_paths, cfg, cfg_ksp,
                epoch + 1, priors)

    return model


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir, cfg.exp_name)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = UNet(out_channels=1,
                 in_channels=3 + (1 if cfg.loc_prior else 0),
                 use_coordconv=cfg.coordconv).to(device)

    transf = make_data_aug(cfg)

    if cfg.aug_method == 'tree':
        dl_train = TreeSetExplorer(pjoin(cfg.in_root,
                                         'Dataset' + cfg.train_dir),
                                   normalization='rescale',
                                   augmentations=transf,
                                   sp_labels_fname='sp_labels.npy',
                                   resize_shape=cfg.in_shape)
        dl_init_noresize = TreeSetExplorer(pjoin(cfg.in_root,
                                                 'Dataset' + cfg.train_dir),
                                           normalization='rescale',
                                           sp_labels_fname='sp_labels.npy')
    else:
        dl_train = SetExplorer(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                               normalization='rescale',
                               augmentations=transf,
                               sp_labels_fname='sp_labels.npy',
                               resize_shape=cfg.in_shape)
        dl_init_noresize = SetExplorer(pjoin(cfg.in_root,
                                             'Dataset' + cfg.train_dir),
                                       normalization='rescale',
                                       sp_labels_fname='sp_labels.npy')

    dl_init = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                     normalization='rescale',
                     resize_shape=cfg.in_shape,
                     sp_labels_fname='sp_labels.npy')

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_train) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_init = DataLoader(dl_init, collate_fn=dl_init.collate_fn)

    dataloader_train = DataLoader(dl_train,
                                  collate_fn=dl_train.collate_fn,
                                  shuffle=True,
                                  batch_size=2)
    dataloader_init_noresize = DataLoader(dl_init_noresize,
                                          collate_fn=dl_train.collate_fn)

    dataloaders = {
        'train': dataloader_train,
        'init_noresize': dataloader_init_noresize,
        'init': dataloader_init,
        'prev': frames_tnsr_brd
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
