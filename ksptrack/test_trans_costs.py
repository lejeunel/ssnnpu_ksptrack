import os
from ksptrack.iterative_ksp import make_link_agent
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
from ksptrack.cfgs import params
import numpy as np
import matplotlib.pyplot as plt
from skimage import (color, io, segmentation, draw)
from ksptrack.utils.link_agent_model import LinkAgentModel


def main(cfg):
    locs2d = utls.readCsv(
        os.path.join(cfg.in_path, cfg.locs_dir, cfg.csv_fname))

    # ---------- Descriptors/superpixel costs
    dm = DataManager(cfg.in_path, cfg.precomp_dir)
    dm.calc_superpix(cfg.slic_compactness, cfg.slic_n_sp)

    link_agent = make_link_agent(dm.labels, cfg)

    if(isinstance(link_agent, LinkAgentModel)):
        print('will use DEC/siam features')
        dm.feats_mode = 'siam'
        #force reload features
        dm.sp_desc_df_ = None

    dm.calc_pm(np.array(link_agent.get_all_entrance_sps(dm.sp_desc_df)),
               cfg.bag_n_feats, cfg.bag_t, cfg.bag_max_depth,
               cfg.bag_max_samples, cfg.bag_jobs)
    labels = dm.labels

    link_agent.update_trans_transform(np.vstack(dm.sp_desc_df['desc'].values),
                                      dm.fg_pm_df['proba'].values,
                                      [cfg.ml_down_thr, cfg.ml_up_thr],
                                      cfg.ml_n_samps,
                                      cfg.lfda_dim,
                                      None,
                                      embedding_type='orthonormalized')

    pm_scores_fg = dm.get_pm_array(mode='foreground')

    trans_probas = np.zeros(labels.shape[:2])
    i_in, j_in = link_agent.get_i_j(locs2d[locs2d['frame'] == cfg.fin])
    label_in = labels[i_in, j_in, cfg.fin]
    for l in np.unique(labels[..., cfg.fout]):
        proba = link_agent.get_proba(cfg.fin, label_in, cfg.fout, l, dm.sp_desc_df)
        trans_probas[labels[..., cfg.fout] == l] = proba

    entrance_probas = np.zeros(labels.shape[:2])
    label_in = labels[i_in, j_in, cfg.fin]
    for l in np.unique(labels[..., cfg.fin]):
        # if(link_agent.is_entrance(cfg.fin, l)):
        proba = link_agent.get_proba(cfg.fin, label_in, cfg.fin, l, dm.sp_desc_df)
        entrance_probas[labels[..., cfg.fin] == l] = proba

    im1 = utls.imread(cfg.frameFileNames[cfg.fin])
    label_cont = segmentation.find_boundaries(labels[..., cfg.fin],
                                              mode='thick')
    aimed_cont = segmentation.find_boundaries(
        labels[..., cfg.fin] == label_in, mode='thick')

    label_cont_im = np.zeros(im1.shape, dtype=np.uint8)
    label_cont_i, label_cont_j = np.where(label_cont)
    label_cont_im[label_cont_i, label_cont_j, :] = 255

    io.imsave('conts.png', label_cont_im)

    rr, cc = draw.circle_perimeter(i_in, j_in,
                                   int(cfg.norm_neighbor * im1.shape[1]))

    im1[rr, cc, 0] = 0
    im1[rr, cc, 1] = 255
    im1[rr, cc, 2] = 0

    im1[aimed_cont, :] = (255, 0, 0)
    im1 = csv.draw2DPoint(locs2d.to_numpy(), cfg.fin, im1, radius=7)

    im2 = utls.imread(cfg.frameFileNames[cfg.fout])
    label_cont = segmentation.find_boundaries(labels[..., cfg.fout],
                                              mode='thick')
    im2[label_cont, :] = (255, 0, 0)

    # cfg.pm_thr = 0.6
    plt.subplot(231)
    plt.imshow(im1)
    plt.title('frame_1. ind: ' + str(cfg.fin))
    plt.subplot(232)
    plt.imshow(im2)
    plt.title('frame_2. ind: ' + str(cfg.fout))
    plt.subplot(233)
    plt.imshow(pm_scores_fg[..., cfg.fin] > cfg.pm_thr)
    plt.title('f1 pm > {}'.format(cfg.pm_thr))
    plt.subplot(234)
    plt.imshow(entrance_probas)
    plt.colorbar()
    plt.title('entrance probas')
    plt.subplot(235)
    plt.imshow(trans_probas)
    plt.colorbar()
    plt.title('trans probas')
    plt.subplot(236)
    plt.imshow(pm_scores_fg[..., cfg.fout])
    plt.title('f2. pm {}'.format(cfg.pm_thr))

    plt.tight_layout()
    path = os.path.join(cfg.run_path, 'test_trans_costs.png')
    print('saving preview to {}'.format(path))
    plt.savefig(path, dpi=400)
    plt.show()


if __name__ == "__main__":
    p = params.get_params()
    p.add('--in-path', required=True)
    p.add('--run-path', required=True)
    p.add('--siam-path', default='')
    p.add('--fin', default=0, type=int)
    p.add('--fout', default=1, type=int)
    cfg = p.parse_args()

    #Make frame file names
    cfg.frameFileNames = utls.get_images(
        os.path.join(cfg.in_path, cfg.frame_dir))
    cfg.precomp_desc_path = os.path.join(cfg.in_path, 'precomp_desc')
    cfg.feats_mode = 'autoenc'

    main(cfg)
