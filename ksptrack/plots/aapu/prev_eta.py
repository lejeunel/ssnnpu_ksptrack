from os.path import join as pjoin
from ksptrack import prev_trans_costs
from ksptrack.cfgs import params
import os
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from ksptrack.iterative_ksp import make_link_agent
from ksptrack.utils import my_utils as utls

if __name__ == "__main__":
    p = params.get_params('../../cfgs')
    p.add('--root-in-path', required=True)
    p.add('--root-run-path', default='')
    p.add('--unlabeled-ratios', nargs='+', type=int, default=[0.1, 0.2, 0.3])
    p.add('--dsets', nargs='+', type=str, default=['02'])
    p.add('--pi-mul', type=float, default=1)
    p.add('--fin', nargs='+', type=int, default=[0])
    p.add('--save-path', default='')
    cfg = p.parse_args()

    cfg.trans = 'lfda'
    cfg.cuda = True
    cfg.bag_t = 600
    cfg.bag_n_feats = 0.1
    cfg.bag_max_depth = 5
    cfg.do_scores = False
    cfg.do_all = False
    cfg.return_dict = True

    dict_path = pjoin(cfg.save_path, 'prev_eta.npy')
    aug_methods = {'none': 'aaPU', 'trees': 'aaPUtrees'}

    if os.path.exists(dict_path):
        print('{} already exists'.format(dict_path))
    else:
        assert len(cfg.dsets) == len(
            cfg.fin), print('give as many dsets as fin')

        dicts = {(dset, m, ur): None
                 for dset, m, ur in zip(cfg.dsets, aug_methods.keys(),
                                        cfg.unlabeled_ratios)}

        for fin, dset in zip(cfg.fin, cfg.dsets):
            cfg.in_path = pjoin(cfg.root_in_path, 'Dataset' + dset)
            # compute bagger's prior using autoencoder's features
            cfg.model_path = pjoin(cfg.root_run_path, 'Dataset' + dset,
                                   'checkpoints', 'init_dec.pth.tar')
            cfg.trans_path = None

            link_agent, desc_df = make_link_agent(cfg)
            probas = utls.calc_pm(
                desc_df, np.array(link_agent.get_all_entrance_sps(desc_df)),
                cfg.bag_n_feats, cfg.bag_t, cfg.bag_max_depth,
                cfg.bag_max_samples, cfg.bag_jobs)
            n_pos = np.concatenate(link_agent.labels_pos).sum()
            n_samps = np.concatenate(link_agent.labels_pos).size
            pos_freq = (probas['proba'] >= 0.5).sum() / n_samps

            print('pos_freq: {}'.format(pos_freq))

            for ur in cfg.unlabeled_ratios:
                cfg.n_augs = ur * (pos_freq * n_samps) - n_pos
                cfg.n_augs = int(max(cfg.n_augs, 0))
                for aug_method in aug_methods.keys():
                    cfg.model_path = pjoin(
                        cfg.root_run_path, 'Dataset' + dset, 'checkpoints',
                        'cp_{}aapu_pimul_{}_ur_{}_pr_bag.pth.tar'.format(
                            '' if aug_method == 'none' else 'tree',
                            float(cfg.pi_mul), ur))
                    cfg.trans_path = ''
                    cfg.use_model_pred = True
                    cfg.aug_method = aug_method
                    cfg.aug_ratio = ur
                    cfg.fin = [fin]

                    dict_ = prev_trans_costs.main(cfg)
                    dicts[(dset, aug_method, ur)] = dict_

        np.save(dict_path, dicts)

    dicts = np.load(dict_path, allow_pickle=True)[()]

    seqs = list(set([k[0] for k in dicts.keys()]))
    methods = list(set([k[1] for k in dicts.keys()]))
    etas = list(set([k[2] for k in dicts.keys()]))
    etas.sort()

    fig = plt.figure()
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(len(methods) * len(seqs), len(etas)),
                     axes_pad=0.02)
    from PIL import Image, ImageDraw, ImageFont

    # get a font
    fnt = ImageFont.truetype("DejaVuSans.ttf", 60)

    pos = 0
    for s in seqs:
        for m in methods:
            im_ = []

            for i, e in enumerate(etas):
                ims = dicts[(s, m, e)]['images'][0]
                ims = np.concatenate((ims['image'], ims['pm_thr']), axis=1)

                if i == 0:
                    text = aug_methods[m]
                    w = ims.shape[0]
                    h = 100
                    # create an image
                    side_label = Image.new("RGB", (w, h), (255, 255, 255))

                    # get a drawing context
                    d = ImageDraw.Draw(side_label)
                    wt, ht = d.textsize(text, font=fnt)
                    d.text(((w - wt) / 2, (h - ht) / 2),
                           text,
                           font=fnt,
                           fill=(0, 0, 0))
                    side_label = np.array(side_label.rotate(90, expand=True))

                    ims = np.concatenate((side_label, ims), axis=1)

                grid[pos].imshow(ims)
                grid[pos].axis('off')
                pos += 1

    fig_path = pjoin(cfg.save_path, 'prev_eta.png')

    titles = [r'$\tilde \eta = {} \eta$'.format(e) for e in etas]
    # titles = ['$\tilde{\eta}={}$'.format(e) for e in etas]

    for pos in range(3):
        grid[pos].set_title(titles[pos], fontsize=8)

    print('saving fig to {}'.format(fig_path))
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
