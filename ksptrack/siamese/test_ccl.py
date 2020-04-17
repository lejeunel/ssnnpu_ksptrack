from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

import clustering as clst
import params
from ksptrack.siamese import utils as utls
from ksptrack.siamese.loader import Loader
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.utils.bagging import calc_bagging
from ksptrack.utils.link_agent_gmm import make_cluster_maps
from ksptrack.prev_trans_costs import colorize
from ksptrack.utils.my_utils import get_pm_array


def run(cfg, model, device, dataloaders, run_path):

    return model


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    alpha=cfg.alpha,
                    backbone=cfg.backbone).to(device)

    dl_single = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                       normalization='rescale',
                       resize_shape=cfg.in_shape)

    dataloader = DataLoader(dl_single, collate_fn=dl_single.collate_fn)

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    path_ = pjoin(run_path, 'checkpoints', 'init_dec.pth.tar')
    print('loading checkpoint {}'.format(path_))
    state_dict = torch.load(path_, map_location=lambda storage, loc: storage)
    model.load_partial(state_dict)

    if (cfg.clf):
        print('changing output of decoder to 1 channel')
        model.dec.autoencoder.to_predictor()

    features, pos_masks = clst.get_features(model, dataloader, device)
    cat_features = np.concatenate(features)
    cat_pos_mask = np.concatenate(pos_masks)
    print('computing probability map')
    probas = calc_bagging(
        cat_features,
        cat_pos_mask,
        # cfg.bag_t,
        30,
        bag_max_depth=cfg.bag_max_depth,
        bag_n_feats=cfg.bag_n_feats,
        n_jobs=1)

    labels = np.rollaxis(dl_single.labels, -1, 0)

    pm_scores_fg = get_pm_array(labels, probas)

    pm_map = colorize(pm_scores_fg[cfg.frame])
    pm_thr_map = colorize(pm_scores_fg[cfg.frame] > 0.5)
    cluster_maps = make_cluster_maps(model, dataloader, device)[cfg.frame]

    probas = torch.from_numpy(probas).to(device)
    n_labels = [np.unique(s['labels']).size for s in dataloader.dataset]
    probas = torch.split(probas, n_labels)
    import pdb
    pdb.set_trace()  ## DEBUG ##
    print('Generating connected components graphs')
    edges_list, subgraphs = utls.make_edges_ccl(model,
                                                dataloader,
                                                device,
                                                probas,
                                                return_subgraphs=True)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)
    p.add('--frame', type=int, required=True)

    cfg = p.parse_args()

    main(cfg)
