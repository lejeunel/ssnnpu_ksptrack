from loader import Loader
from torch.utils.data import DataLoader
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese.distrib_buffer import DistribBuffer
from ksptrack.siamese import im_utils
import numpy as np
from skimage import io
import clustering as clst


def train(cfg, model, device, dataloaders, run_path):
    check_cp_exist = pjoin(run_path, 'checkpoints', 'checkpoint_dec.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Loading.'.format(
            check_cp_exist))
        state_dict = torch.load(check_cp_exist,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=False)

    clusters_prevs_path = pjoin(run_path, 'clusters_prevs')
    if (not os.path.exists(clusters_prevs_path)):
        os.makedirs(clusters_prevs_path)

    init_clusters_path = pjoin(run_path, 'init_clusters.npz')
    init_clusters_prev_path = pjoin(clusters_prevs_path, 'init')

    preds = np.load(init_clusters_path, allow_pickle=True)['preds']
    L = np.load(init_clusters_path, allow_pickle=True)['L']
    L = torch.tensor(L).float().to(device)
    init_clusters = np.load(init_clusters_path, allow_pickle=True)['clusters']
    prev_ims = clst.do_prev_clusters_init(dataloaders['prev'], preds)

    # save initial clusterings to disk
    if (not os.path.exists(init_clusters_prev_path)):
        os.makedirs(init_clusters_prev_path)
        print('saving initial clustering previews...')
        for k, v in prev_ims.items():
            io.imsave(pjoin(init_clusters_prev_path, k), v)

    init_prev = np.vstack([prev_ims[k] for k in prev_ims.keys()])

    writer = SummaryWriter(run_path)

    # can choose order of cycles here
    distrib_buff = DistribBuffer(cfg.tgt_update_period,
                                 thr_assign=cfg.thr_assign)

    writer.add_image('clusters', init_prev, 0, dataformats='HWC')

    init_clusters = torch.tensor(init_clusters,
                                 dtype=torch.float)

    model.dec.set_clusters(init_clusters)
    model.dec.set_transform(L.T)

    path = pjoin(run_path, 'checkpoints')
    print('saving DEC with initial parameters to {}'.format(path))
    utls.save_checkpoint(
        {
            'epoch': -1,
            'model': model
        },
        False,
        fname_cp='init_dec.pth.tar',
        path=path)


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    roi_size=1,
                    roi_scale=cfg.roi_spatial_scale,
                    alpha=cfg.alpha)
    path_cp = pjoin(run_path, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.dec.autoencoder.load_state_dict(state_dict, strict=False)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    _, transf_normal = im_utils.make_data_aug(cfg)

    dl_single = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                       normalization=transf_normal)
    # dl_stack = StackLoader(cfg.batch_size,
    #                        pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
    #                        normalization=transf_normal)
    # dataloader_train = DataLoader(dl_stack,
    #                               collate_fn=dl_stack.collate_fn,
    #                               shuffle=True,
    #                               num_workers=cfg.n_workers)
    dataloader_train = DataLoader(dl_single,
                                  batch_size=2,
                                  collate_fn=dl_single.collate_fn,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_single) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(
        dl_single, frames_tnsr_brd),
                                 collate_fn=dl_single.collate_fn)
    dataloader_buff = DataLoader(dl_single,
                                 collate_fn=dl_single.collate_fn,
                                 num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'buff': dataloader_buff,
        'prev': dataloader_prev
    }

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    train(cfg, model, device, dataloaders, run_path)
    # eval(cfg, model, device, dataloader_buff, run_path)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
