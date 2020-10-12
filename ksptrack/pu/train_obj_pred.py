import os
from os.path import join as pjoin

import numpy as np
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
from ksptrack.pu.modeling.unet import UNet, init_weights_normal, init_weights_unif
from ksptrack.pu.set_explorer import SetExplorer
from ksptrack.pu.tree_set_explorer import TreeSetExplorer
from ksptrack.utils.bagging import calc_bagging
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import numpy as np


def make_pi_curves_data(res_dict, median_filt_size=9):
    true_pi = [f.sum() / f.size for f in res_dict['truths']]
    estim_pi = [(f >= 0.5).sum() / f.size for f in res_dict['outs']]
    estim_pi_filt = medfilt(estim_pi, kernel_size=median_filt_size)

    return {'true': true_pi, 'estim': estim_pi, 'estim_pi_filt': estim_pi_filt}


def write_pi_curves(data, path):

    plt.plot(data['true'], 'b--', label='true')
    plt.plot(data['estim'], 'r-', label='estim')
    plt.plot(data['estim_pi_filt'], 'g-', label='estim_filt')
    plt.grid()
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('positive class prior')
    plt.savefig(path)
    plt.close()


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

        criterion = PULoss(do_ascent=cfg.nnpu_ascent,
                           aug_in_neg=cfg.aug_in_neg)

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
            # with torch.autograd.detect_anomaly():
            res = model(data['image'])

            criterion = make_loss(cfg)
            frames = data['frame_idx']
            pi = [priors[f] for f in frames]
            n_labels = [
                data['pos_labels'][data['pos_labels']['frame'] == f].iloc[0]
                ['n_labels'] for f in frames
            ]
            inp = sp_pool(res['output'], data['labels'])
            inp = torch.split(inp.squeeze(), n_labels)
            target = [
                data['pos_labels'][data['pos_labels']['frame'] == f]
                for f in frames
            ]
            loss = criterion(inp, target, pi=pi, pi_mul=cfg.pi_mul)

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
        'curves_data': pjoin(run_path, 'curves_data')
    }
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    for p in out_paths.values():
        if not os.path.exists(p):
            os.makedirs(p)

    check_cp_exist = pjoin(run_path, cp_fname)
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    path_ = pjoin(os.path.split(run_path)[0], 'autoenc', cp_fname)
    print('loading checkpoint {}'.format(path_))
    state_dict = torch.load(path_, map_location=lambda storage, loc: storage)
    state_dict = {k: v for k, v in state_dict.items() if 'encoder' in k}
    model.load_state_dict(state_dict, strict=False)

    print('computing priors')
    if cfg.true_prior:
        print('using groundtruth')
        res = get_features(model, dataloaders['init'], device)
        truths = res['truths']
        priors = [np.sum(p > 0) / p.size for p in truths]
        pos_freq = np.mean(priors)

    # elif cfg.sec_phase:
    #     assert cfg.pred_init_dir, 'when prior method is not bag, give filename of model'
    #     path_ = pjoin(os.path.split(run_path)[0], cfg.pred_init_dir, cp_fname)
    #     print('loading checkpoint {}'.format(path_))
    #     state_dict = torch.load(path_,
    #                             map_location=lambda storage, loc: storage)
    #     model.load_state_dict(state_dict)
    #     res = get_features(model, dataloaders['init'], device)
    #     probas = res['outs']
    #     priors = [np.sum(p >= 0.5) / p.size for p in probas]
    #     pos_freq = np.mean(priors)
    # else:
    res = get_features(model, dataloaders['init'], device)
    cat_features = np.concatenate(res['feats_bag'])
    cat_pos_mask = np.concatenate(res['labels_pos_mask'])

    probas = calc_bagging(cat_features,
                          cat_pos_mask,
                          600,
                          bag_max_depth=cfg.bag_max_depth,
                          bag_n_feats=cfg.bag_n_feats,
                          n_jobs=1)
    probas = torch.from_numpy(probas)

    pos_freq = ((probas >= 0.5).sum().float() / probas.numel()).item()

    n_labels = [m.shape[0] for m in res['labels_pos_mask']]
    probas = torch.split(probas, n_labels)
    res['outs'] = [p.numpy() for p in probas]
    priors = [((p >= 0.5).sum().float() / p.numel()).item() for p in probas]

    if cfg.pi_filt:
        priors = medfilt(priors, kernel_size=cfg.pi_filt_size)

    res_curves = make_pi_curves_data(res, median_filt_size=cfg.pi_filt_size)

    n_pos = dataloaders['train'].dataset.n_pos
    if (cfg.unlabeled_ratio > 0):
        max_n_augs = cfg.unlabeled_ratio * (
            pos_freq * np.concatenate(res['labels_pos_mask']).shape[0]) - n_pos
        max_n_augs = int(max(max_n_augs, 0))
    else:
        max_n_augs = 0

    first_aug_step = np.round(
        max_n_augs / cfg.epochs_pred).astype(int) * cfg.unlabeled_period
    n_augs = np.linspace(first_aug_step, max_n_augs,
                         cfg.epochs_pred).astype(int)
    print('will add {} augmented samples.'.format(max_n_augs))

    if cfg.pred_init_dir:
        path_ = pjoin(os.path.split(run_path)[0], cfg.pred_init_dir, cp_fname)
        print('loading checkpoint {}'.format(path_))
        state_dict = torch.load(path_,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    path = pjoin(run_path, cp_fname)
    utls.save_checkpoint({
        'epoch': 0,
        'model': model,
        'best_loss': 0,
    }, path)

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
    cfg_ksp.n_augs = 0
    cfg_ksp.aug_method = 'none'
    cfg_ksp.coordconv = cfg.coordconv

    res_prevs = prev_trans_costs.main(cfg_ksp)
    prev_ims = np.concatenate([
        np.concatenate(
            (s['image'], s['pm'], s['pm_thr'], s['entrance']), axis=1)
        for s in res_prevs['images']
    ],
                              axis=0)
    io.imsave(pjoin(out_paths['prevs'], 'ep_{:04d}.png'.format(0)), prev_ims)
    np.savez(pjoin(out_paths['prevs_data'], 'ep_{:04d}.npz'.format(0)),
             **res_prevs)

    np.savez(pjoin(out_paths['curves_data'], 'ep_{:04d}.npz'.format(0)),
             **res_curves)
    write_pi_curves(res_curves,
                    pjoin(out_paths['curves'], 'ep_{:04d}.png'.format(0)))

    writer = SummaryWriter(run_path, flush_secs=1)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_pred))

    if cfg.pred_init_dir:
        pass
    else:
        print('doing first phase')
        print('reinitializing weights')
        model.apply(init_weights_normal)
        offset = -np.log((1 - pos_freq) / pos_freq)
        print('setting objectness prior to {}, p={}'.format(offset, pos_freq))
        model.decoder.output_convolution[0].bias.data.fill_(
            torch.tensor(offset).to(device))

    model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr2 if cfg.sec_phase else cfg.lr1,
                           weight_decay=1e-2)

    lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[cfg.lr_epoch_milestone_0],
        gamma=1. if cfg.sec_phase else cfg.lr_gamma)

    n_epochs = cfg.epochs_pred

    for na, epoch in zip(n_augs, range(n_epochs)):

        n_pos = dataloaders['train'].dataset.n_pos
        print('epoch {}/{}, mode: {}, lr: {:.2e}'.format(
            epoch, n_epochs, 'pred',
            lr_sch.get_last_lr()[0]))

        if (epoch % cfg.unlabeled_period == 0) and na > 0:
            print('augmenting positive set')
            res = get_features(model, dataloaders['init'], device)
            losses = [-np.log(1 - o + 1e-8) for o in res['outs']]
            print('resetting augmented set')
            dataloaders['train'].dataset.make_candidates(losses)
            dataloaders['train'].dataset.reset_augs()
            dataloaders['train'].dataset.augment_positives(na)

            print(dataloaders['train'].dataset)
            writer.add_scalar('n_aug/{}'.format(cfg.exp_name),
                              dataloaders['train'].dataset.n_aug, epoch)
            writer.add_scalar('n_pos/{}'.format(cfg.exp_name),
                              dataloaders['train'].dataset.n_pos, epoch)
            writer.add_scalar('aug_purity/{}'.format(cfg.exp_name),
                              dataloaders['train'].dataset.ratio_purity_augs,
                              epoch)

        train_one_epoch(model,
                        dataloaders['train'],
                        optimizer,
                        device,
                        epoch,
                        lr_sch,
                        cfg,
                        priors=priors,
                        writer=writer)

        # save checkpoint
        if (epoch % cfg.cp_period == 0):
            path = pjoin(run_path, cp_fname)
            utls.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model,
                    'best_loss': best_loss,
                }, path)

        # save previews
        if (epoch % cfg.prev_period == 0) and epoch > 0:
            print('generating previews')

            cfg_ksp.use_model_pred = True
            cfg_ksp.aug_method = cfg.aug_method
            cfg_ksp.n_augs = dataloaders['train'].dataset.n_aug
            cfg_ksp.do_scores = True
            cfg_ksp.sp_labels_fname = 'sp_labels.npy'
            res_prevs = prev_trans_costs.main(cfg_ksp)
            prev_ims = np.concatenate([
                np.concatenate(
                    (s['image'], s['pm'], s['pm_thr'], s['entrance']), axis=1)
                for s in res_prevs['images']
            ],
                                      axis=0)
            io.imsave(pjoin(out_paths['prevs'], 'ep_{:04d}.png'.format(epoch)),
                      prev_ims)
            writer.add_scalar('F1/{}'.format(cfg.exp_name),
                              res_prevs['scores']['f1'], epoch)
            writer.add_scalar('auc/{}'.format(cfg.exp_name),
                              res_prevs['scores']['auc'], epoch)

            io.imsave(pjoin(out_paths['prevs'], 'ep_{:04d}.png'.format(epoch)),
                      prev_ims)

            res = get_features(model, dataloaders['init'], device)
            res_curves = make_pi_curves_data(res,
                                             median_filt_size=cfg.pi_filt_size)
            np.savez(
                pjoin(out_paths['prevs_data'], 'ep_{:04d}.npz'.format(epoch)),
                **res_prevs)

            np.savez(
                pjoin(out_paths['curves_data'], 'ep_{:04d}.npz'.format(epoch)),
                **res_curves)
            write_pi_curves(
                res_curves,
                pjoin(out_paths['curves'], 'ep_{:04d}.png'.format(epoch)))

    return model


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir, cfg.exp_name)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = UNet(out_channels=1, use_coordconv=cfg.coordconv).to(device)

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

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

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
