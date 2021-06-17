import os
from os.path import join as pjoin

import numpy as np
import tqdm
from ksptrack import params
from ksptrack.tracking.ksp_segmentation import make_link_agent
from ksptrack.pu.im_utils import colorize
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import my_utils as utls
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from ksptrack.utils.superpixel_extractor import SuperpixelExtractor
from skimage import draw, io, segmentation, transform
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def main(cfg):
    locs2d = utls.readCsv(
        os.path.join(cfg.in_path, cfg.locs_dir, cfg.csv_fname))

    # ---------- Descriptors/superpixel costs
    spext = SuperpixelExtractor(cfg.in_path,
                                desc_dir=cfg.precomp_dir,
                                compactness=cfg.slic_compactness,
                                n_segments=cfg.slic_n_sp)
    spext.run()

    link_agent, _ = make_link_agent(cfg)

    probas = link_agent.obj_preds
    pm_scores_fg = utls.get_pm_array(link_agent.labels, probas)

    dl = LocPriorDataset(cfg.in_path,
                         normalization='rescale',
                         csv_fname=cfg.csv_fname,
                         sp_labels_fname='sp_labels.npy')

    scores = dict()
    if cfg.do_scores:
        shape = pm_scores_fg.shape[1:]
        truths = np.array([
            transform.resize(s['label/segmentation'],
                             shape,
                             preserve_range=True).astype(np.uint8) for s in dl
        ])
        fpr, tpr, _ = roc_curve(truths.flatten(), pm_scores_fg.flatten())
        precision, recall, _ = precision_recall_curve(
            truths.flatten(),
            pm_scores_fg.flatten() >= 0.5)
        precision = precision[1]
        recall = recall[1]
        nom = 2 * (precision * recall)
        denom = (precision + recall)
        if denom > 0:
            f1 = nom / denom
        else:
            f1 = 0.

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

            truth = dl[fin]['label/segmentation']
            truth_ct = segmentation.find_boundaries(truth, mode='thick')
            im1 = (255 * dl[fin]['image']).astype(np.uint8)
            rr, cc = draw.circle_perimeter(i_in,
                                           j_in,
                                           int(cfg.norm_neighbor_in *
                                               im1.shape[1]),
                                           shape=im1.shape)
            pos_labels = dl[fin]['annotations']

            pos_sps = [
                dl[fin]['labels'].squeeze() == l for l in pos_labels['label']
            ]

            pos_ct = [segmentation.find_boundaries(p) for p in pos_sps]

            for p in pos_ct:
                im1[p, ...] = (0, 255, 0)

            im1[truth_ct, ...] = (255, 0, 0)

            im1[rr, cc, 0] = 0
            im1[rr, cc, 1] = 255
            im1[rr, cc, 2] = 0

            im1 = csv.draw2DPoint(locs2d.to_numpy(), fin, im1, radius=7)
            ims_ = []
            ims_.append(im1)
            ims_.append(colorize(pm_scores_fg[fin]))
            ims_.append(
                colorize((pm_scores_fg[fin] >= cfg.pm_thr).astype(float)))
            ims.append(ims_)

        else:

            im1 = (255 * dl[fin]['image']).astype(np.uint8)
            ims_ = []
            ims_.append(im1)
            ims_.append(colorize(pm_scores_fg[fin]))
            ims_.append(
                colorize((pm_scores_fg[fin] >= cfg.pm_thr).astype(float)))
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
        }
        ims_dicts.append(dict_)
    res['images'] = ims_dicts
    res['scores'] = scores
    return res


if __name__ == "__main__":
    p = params.get_params()
    p.add('--in-path', required=True)
    p.add('--model-path', default='')
    p.add('--trans-path', default='')
    p.add('--use-model-pred', default=False, action='store_true')
    p.add('--trans', default='lfda')
    p.add('--aug-method', default='none', type=str)
    p.add('--do-scores', default=False, action='store_true')
    p.add('--loc-prior', default=False, action='store_true')
    p.add('--fin', nargs='+', type=int, default=[0])
    p.add('--save-path', default='')
    p.add('--aug-df-path', default='')
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
