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



def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DEC(cfg.n_clusters, roi_size=1, roi_scale=cfg.roi_spatial_scale,
                alpha=cfg.alpha)

    path_cp = pjoin(run_path, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.autoencoder.load_state_dict(state_dict)
        model.autoencoder
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    transf, transf_normal = im_utils.make_data_aug(cfg)

    dl_train = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                      normalization=transf_normal)

    dataloader_train = DataLoader(dl_train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl_train.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)
    dataloader_prev = DataLoader(dl_train,
                                 batch_size=1,
                                 collate_fn=dl_train.collate_fn)
    dataloaders = {'train': dataloader_train, 'prev': dataloader_prev}

    check_cp_exist = pjoin(run_path, 'checkpoints', 'checkpoint_dec.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    init_clusters_path = pjoin(run_path, 'init_clusters.npz')
    preds = np.load(init_clusters_path, allow_pickle=True)['preds']
    init_clusters = np.load(init_clusters_path, allow_pickle=True)['clusters']
    init_clusters = torch.tensor(init_clusters,
                                 dtype=torch.float,
                                 requires_grad=True)
    if cfg.cuda:
        init_clusters = init_clusters.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()['assignment.cluster_centers'].copy_(init_clusters)

    model.to(device)

    distrib_buff = DistribBuffer(cfg.tgt_update_period)
    distrib_buff.maybe_update(model, dataloaders['prev'], device)

    criterion_clust = PairwiseConstrainedClustering(cfg.lambda_, cfg.n_edges)
    distrib_buff.maybe_update(model, dataloaders['prev'], device)
    for i, data in enumerate(dataloaders['train']):

        data = utls.batch_to_device(data, device)
        res = model(data)
        distribs, targets = distrib_buff[data['frame_idx']]
        loss, edges_pw = criterion_clust(data['graph'],
                                         res['feats'],
                                         distribs,
                                         targets)
        prev = im_utils.make_grid_samples(data, edges_pw, cfg.n_clusters)
        io.imsave(pjoin(run_path, 'prev_{:04d}.png'.format(i)), prev)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
