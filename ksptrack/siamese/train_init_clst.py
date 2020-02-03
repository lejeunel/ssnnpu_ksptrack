from loader import Loader
import logging
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
import torch.optim as optim
import torch.nn.functional as F
import params
import torch
import datetime
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
import tqdm
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.siamese.modeling.dec import DEC
from ksptrack.siamese.cycle_scheduler import TrainCycleScheduler
from ksptrack.siamese.distrib_buffer import DistribBuffer
from ksptrack.siamese.losses import PairwiseConstrainedClustering, SiameseLoss
from ksptrack.siamese import im_utils
import numpy as np
from skimage import color, io
import glob
import clustering as clst


def train(cfg, model, device, dataloaders, run_path):

    clusters_prevs_path = pjoin(run_path, 'clusters_prevs')
    if (not os.path.exists(clusters_prevs_path)):
        os.makedirs(clusters_prevs_path)

    init_clusters_path = pjoin(run_path, 'init_clusters.npz')
    init_clusters_prev_path = pjoin(clusters_prevs_path, 'init')

    # train initial clustering
    if (not os.path.exists(init_clusters_path)):

        if (not cfg.with_agglo):
            init_clusters, preds = clst.train_kmeans(model,
                                                     dataloaders['buff'],
                                                     device, cfg.n_clusters,
                                                     cfg.with_pck)
        else:
            init_clusters, preds = clst.train_agglo(model, dataloaders['buff'],
                                                    device, cfg.n_clusters,
                                                    cfg.linkage)
        np.savez(init_clusters_path, **{
            'clusters': init_clusters,
            'preds': preds
        })

    preds = np.load(init_clusters_path, allow_pickle=True)['preds']
    init_clusters = np.load(init_clusters_path, allow_pickle=True)['clusters']
    prev_ims = clst.do_prev_clusters_init(dataloaders['prev'], preds)

    # save initial clusterings to disk
    if (not os.path.exists(init_clusters_prev_path)):
        os.makedirs(init_clusters_prev_path)
        print('saving initial clustering previews...')
        for k, v in prev_ims.items():
            io.imsave(pjoin(init_clusters_prev_path, k), v)

    init_prev = np.vstack([
        prev_ims[k] for k in prev_ims.keys()
    ])

    writer = SummaryWriter(run_path)

    writer.add_image('clusters',
                     init_prev,
                     0,
                     dataformats='HWC')

def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    autoenc = DeepLabv3Plus(pretrained=True, embedded_dims=cfg.embedded_dims)
    path_cp = pjoin(run_path, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        autoenc.load_state_dict(state_dict)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return
    model = DEC(autoenc,
                cfg.n_clusters,
                roi_size=1,
                roi_scale=cfg.roi_spatial_scale,
                alpha=cfg.alpha)

    _, transf_normal = im_utils.make_data_aug(cfg)

    dl = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                      normalization=transf_normal)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(dl, frames_tnsr_brd),
                                 collate_fn=dl.collate_fn)
    dataloader_train = DataLoader(dl,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)
    dataloader_buff = DataLoader(dl,
                                 collate_fn=dl.collate_fn,
                                 num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train,
                   'buff': dataloader_buff,
                   'prev': dataloader_prev}

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    train(cfg, model, device, dataloaders, run_path)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
