import os
from ksptrack.iterative_ksp import make_link_agent
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
from ksptrack.cfgs import params
import numpy as np
from skimage import (color, io, segmentation, draw)
from ksptrack.utils.link_agent_gmm import LinkAgentGMM
from ksptrack.siamese.tree_set_explorer import TreeSetExplorer
import matplotlib.pyplot as plt
import tqdm
from os.path import join as pjoin
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
from skimage import transform


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
        probas = utls.calc_pm(
            desc_df, np.array(link_agent.get_all_entrance_sps(desc_df)),
            cfg.bag_n_feats, cfg.bag_t, cfg.bag_max_depth, cfg.bag_max_samples,
            cfg.bag_jobs)
        pm_scores_fg = utls.get_pm_array(link_agent.labels, probas)

    if cfg.use_aug_trees:
        dl = TreeSetExplorer(cfg.in_path,
                             normalization='rescale',
                             csv_fname=cfg.csv_fname,
                             sp_labels_fname='sp_labels_pb.npy')
        if cfg.n_augs > 0:
            dl.make_candidates(probas, dl.labels)
            dl.augment_positives(cfg.n_augs)
    else:
        dl = TreeSetExplorer(cfg.in_path,
                             normalization='rescale',
                             csv_fname=cfg.csv_fname,
                             sp_labels_fname='sp_labels.npy')

    scores = dict()
    if cfg.do_scores:
        shape = pm_scores_fg.shape[1:]
        truths = np.array([
            transform.resize(s['label/segmentation'][..., 0],
                             shape,
                             preserve_range=True).astype(np.uint8) for s in dl
        ])
        fpr, tpr, _ = roc_curve(truths.flatten(), pm_scores_fg.flatten())
        precision, recall, _ = precision_recall_curve(truths.flatten(),
                                                      pm_scores_fg.flatten())
        f1 = (2 * (precision * recall) / (precision + recall)).max()
        auc_ = auc(fpr, tpr)
        scores['f1'] = f1
        scores['auc'] = auc_
        scores['fpr'] = fpr
        scores['tpr'] = tpr

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
            # for l in np.unique(link_agent.labels[fin]):
            for l in range(np.unique(link_agent.labels[fin]).shape[0]):
                proba = link_agent.get_proba(fin, label_in, fin, l, desc_df)
                entrance_probas[link_agent.labels[fin] == l] = proba

            truth = dl[fin]['label/segmentation'][..., 0]
            truth_ct = segmentation.find_boundaries(truth, mode='thick')
            im1 = (255 * dl[fin]['image']).astype(np.uint8)
            rr, cc = draw.circle_perimeter(i_in,
                                           j_in,
                                           int(cfg.norm_neighbor_in *
                                               im1.shape[1]),
                                           shape=im1.shape)
            pos_labels = dl[fin]['pos_labels']
            pos_sps = [
                dl[fin]['labels'].squeeze() == l
                for l in pos_labels[pos_labels['from_aug'] == False]['label']
            ]
            aug_sps = [
                dl[fin]['labels'].squeeze() == l
                for l in pos_labels[pos_labels['from_aug']]['label']
            ]
            pos_ct = [segmentation.find_boundaries(p) for p in pos_sps]
            aug_ct = [segmentation.find_boundaries(p) for p in aug_sps]

            for p in pos_ct:
                im1[p, ...] = (0, 255, 0)
            for p in aug_ct:
                im1[p, ...] = (0, 0, 255)

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
            ims_.append(colorize(entrance_probas))
            ims.append(ims_)

        else:
            im1 = dl[fin]['image_unnormal']

            ims_ = []
            ims_.append(im1)
            ims_.append(colorize(pm_scores_fg[fin]))
            ims_.append(
                colorize((pm_scores_fg[fin] > cfg.pm_thr).astype(float)))
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

    res = dict()
    ims_dicts = []
    for ims_ in ims:
        dict_ = {
            'image': ims_[0],
            'pm': ims_[1],
            'pm_thr': ims_[2],
            'entrance': ims_[3]
        }
        ims_dicts.append(dict_)
    res['images'] = ims_dicts
    res['scores'] = scores
    return res


if __name__ == "__main__":
    p = params.get_params()
    p.add('--in-path', required=True)
    p.add('--siam-path', default='')
    p.add('--use-siam-pred', default=False, action='store_true')
    p.add('--use-siam-trans', default=False, action='store_true')
    p.add('--use-aug-trees', default=False, action='store_true')
    p.add('--do-scores', default=False, action='store_true')
    p.add('--n-augs', type=int, default=0)
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
