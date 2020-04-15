import os
from ksptrack.iterative_ksp import make_link_agent
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
from ksptrack.cfgs import params
import numpy as np
from skimage import (color, io, segmentation, draw)
from ksptrack.utils.link_agent_gmm import LinkAgentGMM
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
import matplotlib.pyplot as plt
import tqdm
from os.path import join as pjoin


def colorize(map_):
    cmap = plt.get_cmap('viridis')
    map_colorized = (cmap(map_)[..., :3] * 255).astype(np.uint8)

    return map_colorized


def main(cfg):
    locs2d = utls.readCsv(
        os.path.join(cfg.in_path, cfg.locs_dir, cfg.csv_fname))

    # ---------- Descriptors/superpixel costs
    dm = DataManager(cfg.in_path, cfg.precomp_dir)
    dm.calc_superpix(cfg.slic_compactness, cfg.slic_n_sp)

    link_agent, desc_df = make_link_agent(cfg)

    if (cfg.use_siam_pred):
        print('will use DEC/siam objectness probabilities')
        probas = link_agent.obj_preds
        pm_scores_fg = utls.get_pm_array(link_agent.labels, probas)
    else:
        pm = utls.calc_pm(desc_df,
                          np.array(link_agent.get_all_entrance_sps(desc_df)),
                          cfg.bag_n_feats, cfg.bag_t, cfg.bag_max_depth,
                          cfg.bag_max_samples, cfg.bag_jobs)
        pm_scores_fg = utls.get_pm_array(link_agent.labels, pm)

    dl = LocPriorDataset(cfg.in_path,
                         resize_shape=512,
                         normalization='rescale',
                         csv_fname=cfg.csv_fname)

    cluster_maps = link_agent.make_cluster_maps()

    if (cfg.do_all):
        cfg.fin = np.arange(len(dl))

    ims = []
    pbar = tqdm.tqdm(total=len(cfg.fin))
    for fin in cfg.fin:

        loc = locs2d[locs2d['frame'] == fin]
        if (loc.shape[0] > 0):
            i_in, j_in = link_agent.get_i_j(loc.iloc[0])

            entrance_probas = np.zeros(link_agent.labels.shape[1:])
            label_in = link_agent.labels[fin, i_in, j_in]
            for l in np.unique(link_agent.labels[fin]):
                proba = link_agent.get_proba(fin, label_in, fin, l, desc_df)
                entrance_probas[link_agent.labels[fin] == l] = proba

            truth = dl[fin]['label/segmentation'][..., 0]
            truth_ct = segmentation.find_boundaries(truth, mode='thick')
            im1 = dl[fin]['image_unnormal']
            rr, cc = draw.circle_perimeter(i_in,
                                           j_in,
                                           int(cfg.norm_neighbor_in *
                                               im1.shape[1]),
                                           shape=im1.shape)

            im1[truth_ct, ...] = (255, 0, 0)

            im1[rr, cc, 0] = 0
            im1[rr, cc, 1] = 255
            im1[rr, cc, 2] = 0

            im1 = csv.draw2DPoint(locs2d.to_numpy(), fin, im1, radius=7)
            ims_ = []
            ims_.append(im1)
            ims_.append(colorize(pm_scores_fg[fin]))
            ims_.append(
                colorize((pm_scores_fg[fin] > cfg.pm_thr).astype(float)))
            ims_.append(cluster_maps[fin])
            ims_.append(colorize(entrance_probas))
            ims.append(ims_)

        else:
            im1 = dl[fin]['image_unnormal']

            ims_ = []
            ims_.append(im1)
            ims_.append(colorize(pm_scores_fg[fin]))
            ims_.append(
                colorize((pm_scores_fg[fin] > cfg.pm_thr).astype(float)))
            ims_.append(cluster_maps[fin])
            ims_.append(colorize(np.zeros_like(pm_scores_fg[fin])))
            ims.append(ims_)

        pbar.update(1)
    pbar.close()

    if (cfg.do_all):
        print('will save all to {}'.format(cfg.save_path))
        if (not os.path.exists(cfg.save_path)):
            os.makedirs(cfg.save_path)
        pbar = tqdm.tqdm(total=len(ims))
        for i, im in enumerate(ims):
            io.imsave(pjoin(cfg.save_path, 'im_{:04d}.png'.format(i)),
                      np.concatenate(im, axis=1))
            pbar.update(1)
        pbar.close()

    if (cfg.return_dict):
        ims_dicts = []
        for ims_ in ims:
            dict_ = {
                'image': ims_[0],
                'pm': ims_[1],
                'pm_thr': ims_[2],
                'clusters': ims_[3],
                'entrance': ims_[4]
            }
            ims_dicts.append(dict_)
        return ims_dicts

    return np.concatenate([np.concatenate(im, axis=1) for im in ims], axis=0)


if __name__ == "__main__":
    p = params.get_params()
    p.add('--in-path', required=True)
    p.add('--siam-path', default='')
    p.add('--use-siam-pred', default=False, action='store_true')
    p.add('--use-siam-trans', default=False, action='store_true')
    p.add('--fin', nargs='+', type=int, default=[0])
    p.add('--save-path', default='')
    p.add('--do-all', default=False, action='store_true')
    p.add('--return-dict', default=False, action='store_true')
    cfg = p.parse_args()

    #Make frame file names
    # cfg.frameFileNames = utls.get_images(
    #     os.path.join(cfg.in_path, cfg.frame_dir))
    # cfg.precomp_desc_path = os.path.join(cfg.in_path, 'precomp_desc')
    # cfg.feats_mode = 'autoenc'

    ims = main(cfg)

    if (not cfg.do_all):
        print('saving image to {}'.format(cfg.save_path))
        io.imsave(cfg.save_path, ims)
