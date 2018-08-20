import logging
import numpy as np
from ksptrack.utils import my_utils as utls
import progressbar
import os
import pickle as pk
import matplotlib.pyplot as plt

class SuperpixelManager:

    """
    Builds dictionaries to decide which SP connects to which (kind of filtering of candidates)
    """
    def __init__(self,
                 dataset,
                 conf,
                 direction='forward',
                 with_flow = False):

        self.dataset = dataset
        self.labels = dataset.get_labels()
        self.c_loc = dataset.centroids_loc
        self.with_flow = with_flow
        self.logger = logging.getLogger('SuperpixelManager')
        self.norm_dist_t = 0.2 #Preliminary threshold (normalized coord.)
        self.direction = direction
        #self.dict_ = self.make_init_dicts()
        self.conf = conf

    def make_dicts(self):
        #self.dict_init = self.make_init_dicts()
        if(self.with_flow):
            self.dict_ = self.make_hoof_dict()
        else:
            self.dict_ = self.apply_overlap_constraint()


    def make_init_dicts(self):
        """
        Makes a first "gross" filtering of transition dictionary.
        Uses a threshold parameter self.norm_dist_t chosen large enough.
        """

        file_dict = os.path.join(self.conf.dataOutDir , 'init_constraint_' + self.direction + '.p')
        if(not os.path.exists(file_dict)):
            f = self.c_loc['frame']
            s = self.c_loc['sp_label']
            dict_ = {key:[] for key in zip(f, s)}

            self.c_loc_sort_x = self.c_loc.copy()
            self.c_loc_sort_y = self.c_loc.copy()

            frames = np.unique(f)
            if(self.direction == 'backward'):
                frames = frames[::-1]

            frames = [(frames[i],frames[i+1]) for i in range(frames.shape[0]-1)]

            self.logger.info('Building SP links dictionary')
            with progressbar.ProgressBar(maxval=len(frames)) as bar:
                for i in range(len(frames)):
                    bar.update(i)
                    df1 = self.c_loc.loc[self.c_loc['frame'] == frames[i][0]]
                    df2 = self.c_loc.loc[self.c_loc['frame'] == frames[i][1]]
                    combined = utls.df_crossjoin(df1,df2, suffixes=('_in', '_out'))
                    combined['diff_x'] = combined['pos_norm_x_in'] - combined['pos_norm_x_out']
                    combined['diff_y'] = combined['pos_norm_y_in'] - combined['pos_norm_y_out']
                    combined['dist'] = np.sqrt(combined['diff_x']**2 + combined['diff_y']**2)
                    combined = combined[combined['dist'] < self.norm_dist_t]
                    for r in combined.itertuples(index=False):
                        key = (r[0],r[1])
                        val = (r[4],r[5])
                        dict_[key].append(val)
            self.logger.info('Saving init dictionary to ' + file_dict)
            with open(file_dict, 'wb') as f:
                pk.dump(dict_, f, pk.HIGHEST_PROTOCOL)
        else:
            self.logger.info('Loading init dictionary... (delete to re-run)')
            with open(file_dict, 'rb') as f:
                return pk.load(f)
        self.dict_init = dict_
        return dict_

    def apply_overlap_constraint(self):
        """
        Makes a first "gross" filtering of transition dictionary.
        Uses a threshold parameter self.norm_dist_t chosen large enough.
        """

        file_dict = os.path.join(self.conf.precomp_desc_path,
                                 'overlap_constraint_' + self.direction + '.p')
        if(not os.path.exists(file_dict)):
            f = self.c_loc['frame']
            s = self.c_loc['sp_label']

            frames = np.unique(f)
            if(self.direction ==  'backward'):
                frames = frames[::-1]

            frames_tup = [(frames[i],frames[i+1])
                      for i in range(frames.shape[0]-1)]
            dict_ = dict()

            self.logger.info('Building SP overlap dictionary')
            with progressbar.ProgressBar(maxval=len(frames_tup)) as bar:
                for i in range(len(frames_tup)):
                    bar.update(i)

                    # Find overlapping labels between consecutive frames
                    l_0 = self.labels[..., frames_tup[i][0]][..., np.newaxis]
                    l_1 = self.labels[..., frames_tup[i][1]][..., np.newaxis]
                    concat_ = np.concatenate((l_0,l_1), axis=-1)
                    concat_ = concat_.reshape((-1,2))
                    ovl = np.asarray(list(set(list(map(tuple, concat_)))))
                    ovl_list = list()
                    for l0 in np.unique(ovl[:, 0]):
                        ovl_ = [l0, ovl[ovl[:,0] == l0, 1].tolist()]
                        ovl_list.append(ovl_)
                    f0 = frames_tup[i][0]
                    f1 = frames_tup[i][1]
                    dict_.update({(f0, ovl_list[i][0]):(f1, ovl_list[i][1])
                                  for i in range(len(ovl_list))})

            dict_.update({(frames[-1], l): (frames[-1], []) for l in
                          np.unique(self.labels[..., frames[-1]])})

            self.logger.info('Saving overlap dictionary to ' + file_dict)
            with open(file_dict, 'wb') as f:
                pk.dump(dict_, f, pk.HIGHEST_PROTOCOL)
        else:
            self.logger.info('Loading overlap dictionary... (delete to re-run)')
            with open(file_dict, 'rb') as f:
                return pk.load(f)
        self.dict_init = dict_
        return dict_

    def make_hoof_dict(self):
        """
        Associates to dictionary the intersection of HOOF
        """

        self.dict_over = self.apply_overlap_constraint()
        dict_ = dict()

        file_dict = os.path.join(self.conf.precomp_desc_path,
                                 'hoof_constraint_{}.p'.format(self.direction))

        frames = np.unique(self.c_loc['frame'])
        # Compute histograms
        if(not os.path.exists(file_dict)):
            if(self.direction == 'forward'):
                keys = ['fvx', 'fvy']
            else:
                keys = ['bvx', 'bvy']

            hoof = dict()
            bins_hoof = np.linspace(-np.pi,np.pi,self.conf.n_bins_hoof+1)

            self.logger.info('Getting optical flows')
            flows = self.dataset.get_flows()
            self.logger.info('Computing HOOF')

            # duplicate flows at limits
            #if(self.direction == 'forward'):
            fx = np.concatenate((flows[keys[0]],
                                flows[keys[0]][..., -1][..., np.newaxis]),
                                axis=-1)
            fy = np.concatenate((flows[keys[1]],
                                flows[keys[1]][..., -1][..., np.newaxis]),
                                axis=-1)
            #else:
            #    fx = np.concatenate((flows[keys[0]][..., -1][..., np.newaxis],
            #                         flows[keys[0]]),
            #                        axis=-1)
            #    fy = np.concatenate((flows[keys[1]][..., -1][..., np.newaxis],
            #                         flows[keys[1]]),
            #                        axis=-1)

            with progressbar.ProgressBar(maxval=len(frames)) as bar:
                for f, p in zip(frames, range(len(frames))):
                    hoof[f] = dict()
                    bar.update(p)
                    unq_labels = np.unique(self.labels[..., f])
                    bins_label = range(unq_labels.size + 1)
                    angle = np.arctan2(fx[..., p],
                                       fy[..., p])
                    norm = np.linalg.norm(
                        np.concatenate((fx[..., p][...,np.newaxis],
                                        fy[..., p][...,np.newaxis]),
                                                         axis=-1),
                        axis=-1).ravel()

                    # Get superpixel indices for each label
                    l_mask = [self.labels[..., f].ravel() == l
                            for l in unq_labels]

                    # Get angle-bins for each pixel
                    b_angles = np.digitize(angle.ravel(), bins_hoof).ravel()

                    # Get angle-bins indices for each pixel
                    b_mask = [b_angles.ravel() == b
                              for b in range(1, len(bins_hoof))]

                    # Sum norms for each bin and each label
                    hoof_ = np.asarray([[np.sum(norm[l_ & b_])
                             for b_ in b_mask]
                             for l_ in l_mask])

                    # Normalize w.r.t. L1-norm
                    l1_norm = np.sum(hoof_, axis = 1).reshape(
                        (unq_labels.size, 1))
                    hoof_ = np.nan_to_num(hoof_/l1_norm)

                    # Store HOOF in dictionary
                    hoof[f] = {unq_labels[i]: hoof_[i, :]
                               for i in range(len(unq_labels))}

            self.logger.info('Computing HOOF intersections')
            hoof_inter = dict()
            n_keys = len(self.dict_over.keys())
            with progressbar.ProgressBar(maxval=n_keys) as bar:
                for k, i in zip(self.dict_over.keys(),
                             range(len(self.dict_over.keys()))):
                    bar.update(i)
                    hoof_inter[k] = dict()
                    tmp_list = list()
                    for c in self.dict_over[k][1]: # candidates
                        start_f = k[0]
                        end_f = self.dict_over[k][0]
                        start_s = k[1] # start sp_label
                        end_s = c
                        inter_hoof = utls.hist_inter(
                            hoof[start_f][start_s],
                            hoof[end_f][end_s])
                        tmp_list.append((end_f,end_s,inter_hoof))
                    hoof_inter[k] = tmp_list

            self.logger.info('Saving dictionary to ' + file_dict)
            with open(file_dict, 'wb') as f:
                pk.dump(hoof_inter, f, pk.HIGHEST_PROTOCOL)
            self.dict_hoof = hoof_inter
            return self.dict_hoof
        else:
            self.logger.info('Loading dictionary... (delete to re-run)')
            with open(file_dict, 'rb') as f:
                self.dict_hoof = f
                return pk.load(f)
