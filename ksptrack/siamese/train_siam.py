from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from ksptrack.siamese.loader import Loader
from torch.utils.data import DataLoader
import torch.optim as optim
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese import utils as utls
from ksptrack.siamese import im_utils
from ksptrack.siamese.losses import LabelKLPairwiseLoss, LDALoss
import numpy as np
from skimage import io
import clustering as clst
from ksptrack.utils.bagging import calc_bagging
import networkx as nx
from ksptrack.models.losses import PriorMSE
from ksptrack import prev_trans_costs
from ksptrack.cfgs import params as params_ksp
from ksptrack.siamese.distrib_buffer import DistribBuffer
from torch.nn.functional import sigmoid, tanh


def get_keypoints_on_batch(data):
    kp_labels = []
    max_node = 0
    n_nodes = [g.number_of_nodes() for g in data['graph']]
    for labels, n_nodes_ in zip(data['label_keypoints'], n_nodes):
        for l in labels:
            l += max_node
            kp_labels.append(l)
        max_node += n_nodes_ + 1

    return kp_labels


def train_one_epoch(model, dataloaders, optimizers, device,
                    distrib_buff,
                    lr_sch, cfg, all_edges_nn=None, probas=None):

    model.train()

    running_loss = 0.0
    running_clst = 0.0
    running_recons = 0.0
    running_gcn = 0.0

    # criterion_recons = PriorMSE()
    criterion_recons = torch.nn.MSELoss()
    criterion_clst = torch.nn.KLDivLoss(reduction='mean')
    criterion_gcn = torch.nn.KLDivLoss(reduction='mean')
    # criterion_lda = LDALoss(thr=cfg.ml_up_thr)

    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            for k in optimizers.keys():
                optimizers[k].zero_grad()

            # nodes_color = torch.cat([[g.nodes[n]['mean color']
            #                           for n in g.nodes()]
            #                          for g in data['graph']]).to(device)

            edges_nn = None
            nodes_color = None
            probas_ = None

            _, targets = distrib_buff[data['frame_idx']]

            if(cfg.pw):
                probas_ = torch.cat([probas[i] for i in data['frame_idx']])
                edges_nn = utls.combine_nn_edges(
                    [all_edges_nn[i] for i in data['frame_idx']])
                res = model(data, edges_nn=edges_nn,
                            probas=probas_,
                            nodes_color=nodes_color,
                            thrs=[cfg.ml_down_thr,
                                  cfg.ml_up_thr])
                loss_gcn = criterion_gcn(res['clusters_gcn'],
                                         targets.to(res['clusters_gcn']))
            else:
                res = model(data)

            loss_recons = criterion_recons(sigmoid(res['output']),
                                           data['image'])
            loss = cfg.gamma * loss_recons

            # loss_lda = criterion_lda(res['proj_pooled_aspp_feats'],
            #                          probas_)
            # loss += loss_lda

            loss_clst = criterion_clst(res['clusters'], targets.to(res['clusters']))

            loss += loss_clst

            if(cfg.pw):
                loss += cfg.beta * loss_gcn

            loss.backward()

            for k in optimizers.keys():
                optimizers[k].step()

            for k in lr_sch.keys():
                lr_sch[k].step()

        running_loss += loss.cpu().detach().numpy()
        running_recons += loss_recons.cpu().detach().numpy()
        running_clst += loss_clst.cpu().detach().numpy()
        if(cfg.pw):
            running_gcn += loss_gcn.cpu().detach().numpy()
        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        pbar.set_description('lss {:.6f}'.format(loss_))
        pbar.update(1)

    pbar.close()

    loss_recons = running_recons / (cfg.batch_size * len(dataloaders['train']))
    loss_gcn = running_gcn / (cfg.batch_size * len(dataloaders['train']))
    loss_clst = running_clst / (cfg.batch_size * len(dataloaders['train']))

    out = {'loss': loss_,
           'loss_gcn': loss_gcn,
           'loss_clst': loss_clst,
           'loss_recons': loss_recons}

    return out


def train(cfg, model, device, dataloaders, run_path):

    if (cfg.pw):
        cp_fname = 'checkpoint_siam_pw.pth.tar'
        best_cp_fname = 'best_siam_pw.pth.tar'
        rags_prevs_path = pjoin(run_path, 'prevs_pw')
    else:
        cfg.beta = 0.
        cp_fname = 'checkpoint_siam.pth.tar'
        best_cp_fname = 'best_siam.pth.tar'
        rags_prevs_path = pjoin(run_path, 'prevs')

    path_ = pjoin(run_path, 'checkpoints', 'init_dec.pth.tar')
    print('loading checkpoint {}'.format(path_))
    state_dict = torch.load(path_,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    check_cp_exist = pjoin(run_path, 'checkpoints', best_cp_fname)
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    features, pos_masks = clst.get_features(model, dataloaders['all_prev'],
                                            device)
    cat_features = np.concatenate(features)
    cat_pos_mask = np.concatenate(pos_masks)
    print('computing probability map')
    probas = calc_bagging(cat_features,
                          cat_pos_mask,
                          cfg.bag_t,
                          bag_max_depth=cfg.bag_max_depth,
                          bag_n_feats=cfg.bag_n_feats,
                          n_jobs=1)
    probas = torch.from_numpy(probas).to(device)
    n_labels = [
        np.unique(s['labels']).size for s in dataloaders['all_prev'].dataset
    ]
    probas = torch.split(probas, n_labels)

    if (not os.path.exists(rags_prevs_path)):
        os.makedirs(rags_prevs_path)

    p_ksp = params_ksp.get_params('../cfgs')
    p_ksp.add('--siam-path', default='')
    p_ksp.add('--in-path', default='')
    p_ksp.add('--do-all', default=False)
    p_ksp.add('--return-dict', default=False)
    p_ksp.add('--fin', nargs='+')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]
    cfg_ksp.bag_t = 300
    cfg_ksp.bag_n_feats = cfg.bag_n_feats
    cfg_ksp.bag_max_depth = cfg.bag_max_depth
    cfg_ksp.siam_path = pjoin(run_path, 'checkpoints', 'init_dec.pth.tar')
    cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
    cfg_ksp.precomp_desc_path = pjoin(cfg_ksp.in_path, 'precomp_desc')
    cfg_ksp.fin = [s['frame_idx'] for s in dataloaders['prev'].dataset]
    prev_ims = prev_trans_costs.main(cfg_ksp)

    print('generating previews to {}'.format(rags_prevs_path))

    io.imsave(pjoin(rags_prevs_path, 'ep_0000.png'), prev_ims)

    writer = SummaryWriter(run_path)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_dist))

    model.to(device)

    optimizers = {
        'gcn':
        optim.SGD(params=[{
            'params': model.gcns.parameters(),
            'lr': cfg.lr_dist,
        }],
                  momentum=cfg.momentum,
                  weight_decay=cfg.decay),
        'feats':
        optim.SGD(params=[{
            'params': model.dec.autoencoder.parameters(),
            'lr': cfg.lr_autoenc,
        }],
                  momentum=cfg.momentum,
                  weight_decay=cfg.decay),
        'L':
        optim.SGD(params=[{
            'params': model.dec.transform.parameters(),
            'lr': cfg.lr_dist,
        }]),
        'assign':
        optim.SGD(params=[{
            'params': model.dec.assignment.parameters(),
            'lr': cfg.lr_dist,
        }])
    }

    lr_sch = {'feats': torch.optim.lr_scheduler.ExponentialLR(optimizers['feats'],
                                                              0.99),
              # 'feats_pw0': torch.optim.lr_scheduler.ExponentialLR(optimizers['feats_pw0'],
              #                                             0.99),
              # 'feats_pw1': torch.optim.lr_scheduler.ExponentialLR(optimizers['feats_pw1'],
              #                                             0.99),
              'L': torch.optim.lr_scheduler.ExponentialLR(optimizers['L'],
                                                          0.99),
              'assign': torch.optim.lr_scheduler.ExponentialLR(optimizers['assign'],
                                                               0.99)}
    distrib_buff = DistribBuffer(cfg.tgt_update_period,
                                 thr_assign=cfg.thr_assign)


    if(cfg.pw):
        all_edges_nn = [
            utls.make_single_graph_nn_edges(s['graph'], device, cfg.nn_radius)
            for s in dataloaders['all_prev'].dataset
        ]
    else:
        all_edges_nn = None

    for epoch in range(1, cfg.epochs_dist + 1):
        if (distrib_buff.converged):
            print('clustering assignment hit threshold. Ending training.')
            break

        print('epoch {}/{}'.format(epoch, cfg.epochs_dist))
        for phase in ['train', 'prev']:
            if phase == 'train':
                distrib_buff.maybe_update(model, dataloaders['all_prev'],
                                          device,
                                          thrs=[cfg.ml_down_thr, cfg.ml_up_thr])

                res = train_one_epoch(model, dataloaders,
                                      optimizers, device,
                                      distrib_buff,
                                      lr_sch, cfg,
                                      probas=probas,
                                      all_edges_nn=all_edges_nn)

                if(epoch < cfg.epochs_dist):
                    distrib_buff.inc_epoch()

                # write losses to tensorboard
                for k, v in res.items():
                    writer.add_scalar(k, v, epoch)

                if ((epoch % cfg.proba_update_period == 0) and cfg.pw):
                    features, pos_masks = clst.get_features(model,
                                                            dataloaders['all_prev'],
                                                            device,
                                                            all_edges_nn=all_edges_nn,
                                                            probas=probas,
                                                            feat_field='pooled_aspp_gcn_feats')
                    cat_features = np.concatenate(features)
                    cat_pos_mask = np.concatenate(pos_masks)
                    print('computing probability map')
                    probas = calc_bagging(cat_features,
                                          cat_pos_mask,
                                          cfg.bag_t,
                                          bag_n_feats=cfg.bag_n_feats,
                                          bag_max_depth=cfg.bag_max_depth,
                                          n_jobs=1)
                    probas = torch.from_numpy(probas).to(device)
                    n_labels = [
                        np.unique(s['labels']).size for s in dataloaders['all_prev'].dataset
                    ]
                    probas = torch.split(probas, n_labels)

                # save checkpoint
                if (epoch % cfg.cp_period == 0):
                    is_best = False
                    if (res['loss'] < best_loss):
                        is_best = True
                        best_loss = res['loss']
                    path = pjoin(run_path, 'checkpoints')
                    utls.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'model': model,
                            'best_loss': best_loss,
                        },
                        is_best,
                        fname_cp=cp_fname,
                        fname_bm=best_cp_fname,
                        path=path)

            else:

                # save previews
                if (epoch % cfg.prev_period == 0):
                    out_path = rags_prevs_path
                                     
                    print('generating previews to {}'.format(out_path))

                    cfg_ksp.siam_path = pjoin(run_path, 'checkpoints', cp_fname)
                    prev_ims = prev_trans_costs.main(cfg_ksp)
                    io.imsave(pjoin(out_path, 'ep_{:04d}.png'.format(epoch)), prev_ims)

                    # write previews to tensorboard
                    # prev_ims_pt = np.vstack([im for im in prev_ims.values()])
                    # writer.add_image('rags',
                    #                  prev_ims_pt,
                    #                  epoch,
                    #                  dataformats='HWC')


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = Siamese(cfg.embedded_dims,
                    cfg.n_clusters,
                    roi_size=1,
                    roi_scale=cfg.roi_spatial_scale,
                    alpha=cfg.alpha).to(device)

    _, transf_normal = im_utils.make_data_aug(cfg)

    dl_single = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                       normalization=transf_normal)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_single) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(
        dl_single, frames_tnsr_brd),
                                 collate_fn=dl_single.collate_fn)

    dataloader_all_prev = DataLoader(dl_single,
                                     collate_fn=dl_single.collate_fn)

    dataloader_train = DataLoader(dl_single,
                                  collate_fn=dl_single.collate_fn,
                                  batch_size=cfg.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'all_prev': dataloader_all_prev,
        'prev': dataloader_prev
    }

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    train(cfg, model, device, dataloaders, run_path)

    # prev_ims = clst.do_prev_rags(model, device,
    #                              dataloaders['all_prev'],
    #                              couple_graphs, L)

    # save last clusterings to disk
    # last_rags_prev_path = pjoin(run_path, 'rags_prevs', 'last')
    # if (not os.path.exists(last_rags_prev_path)):
    #     os.makedirs(last_rags_prev_path)
    #     print('saving last rags previews...')
    #     for k, v in prev_ims.items():
    #         io.imsave(pjoin(last_rags_prev_path, k), v)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
