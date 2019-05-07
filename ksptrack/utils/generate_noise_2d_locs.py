import os
import numpy as np
import datetime
from labeling.cfgs import cfg_unet, cfg
from labeling.utils import my_utils as utls
from labeling.utils import learning_dataset
from labeling.utils import csv_utils as csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.morphology as morphology

extra_cfg = dict()

all_datasets = ['Dataset00']
ratio_noisy = [.05, .1, .2, .4, .5]

csv_columns = ['frame', 'time', 'visible', 'x', 'y']

#mode = 'all_bg' # Sample uniformly on background
#neigh_dists = [0.]
mode = 'neigh'# Sample uniformly on background (neighborhood of object)
neigh_dists = [.05, .1]

# Run KSP on all seqs with first gaze-set and make prediction experiment
for i in range(len(all_datasets)):
    set_confs = []
    extra_cfg['ds_dir'] = all_datasets[i]
    extra_cfg['seq_type'] = cfg.datasetdir_to_type(extra_cfg['ds_dir'])
    cfg_dict = cfg.cfg()
    cfg_dict.update(extra_cfg)
    conf = cfg.dict_to_munch(cfg_dict)

    print("dset: " + all_datasets[i])
    #for k in [1, 2, 3, 4, 5]:
    for dist in neigh_dists:
        for k in [1]:
            for r in ratio_noisy:
                candidates = list()
                save_out_csv = ('video{}_{}'
                                '_ratio_{}_dist_{}.csv').format(k,
                                                        mode,
                                                        int(100*r),
                                                        int(100*dist))
                save_out_cands = ('candidates{}_{}'
                     '_ratio_{}_dist_{}.npz').format(k,
                                                    mode,
                                                    int(100*r),
                                                    int(100*dist))
                save_out_path = os.path.join(conf.root_path,
                                            conf.ds_dir,
                                            conf.locs_dir,
                                            save_out_csv)
                save_out_path_cands = os.path.join(conf.root_path,
                                                conf.ds_dir,
                                                conf.locs_dir,
                                                save_out_cands)

                extra_cfg['csvFileName_fg'] = 'video' + str(k) + '.csv'

                orig_2dlocs = utls.readCsv(os.path.join(conf.root_path,
                                                        conf.ds_dir,
                                                        conf.locs_dir,
                                                        conf.csvFileName_fg))
                noised_2dlocs = orig_2dlocs.copy()

                conf.frameFileNames = utls.makeFrameFileNames(
                    conf.frame_prefix, conf.frameDigits, conf.frameDir,
                    conf.root_path, conf.ds_dir, conf.frame_extension)


                imgs = [utls.imread(f) for f in conf.frameFileNames]
                imgs = np.asarray(imgs).transpose((1,2,3,0))
                lset = learning_dataset.LearningDataset(conf)
                gts = lset.gt
                candidates = gts.copy()

                inds_to_noise = np.random.choice(np.arange(imgs.shape[-1]),
                                                size=int(imgs.shape[-1]*r),
                                                replace=False)
                for ind_to_noise in inds_to_noise:
                    if(mode == 'all_bg'):
                        cand_ = gts[..., ind_to_noise] == 0
                    elif(mode == 'neigh'):
                        import pdb; pdb.set_trace()
                        dilated_gt = morphology.binary_dilation(
                            gts[..., ind_to_noise],
                            selem=morphology.disk(int(dist*gts.shape[1])))
                        cand_ = np.logical_xor(dilated_gt,
                                                    gts[..., ind_to_noise])

                    candidates[..., ind_to_noise] = cand_
                    ind_bg = np.where(cand_.ravel())[0]
                    ind_bg = np.random.choice(ind_bg)
                    loc_ = np.unravel_index(ind_bg, imgs.shape[0:2])
                    noised_loc = csv.pix2Norm(loc_[1],
                                            loc_[0],
                                            imgs.shape[1],
                                            imgs.shape[0])
                    noised_loc = np.around(np.asarray(noised_loc), decimals=6)

                    noised_2dlocs[ind_to_noise, 3:5] = noised_loc
                # Save noised coordinates
                #print('Saving to {}'.format(save_out_path))
                #df = pd.DataFrame(data=noised_2dlocs,
                #                columns=csv_columns)
                #df.to_csv(path_or_buf=save_out_path)

                #print('Saving to {}'.format(save_out_cands))
                #np.savez(save_out_path_cands, **{'candidates': candidates})

im_preview = imgs[..., ind_to_noise]
im_preview = csv.draw2DPoint(orig_2dlocs,
                                ind_to_noise,
                                im_preview,
                                radius=10)
im_preview = csv.draw2DPoint(noised_2dlocs,
                                ind_to_noise,
                                im_preview,
                                color=(0,0,255),
                                radius=10)
plt.imshow(im_preview);
plt.show()
