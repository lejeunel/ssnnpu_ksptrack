import scipy
from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import superpixel_utils as spix
from ksptrack.utils import my_utils as utls
from ksptrack.utils import bagging as bag
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import superpixel_extractor as svx
from ksptrack.utils import optical_flow_extractor as oflowx
from ksptrack.cfgs import cfg
import pickle as pk
from progressbar import ProgressBar as pbar
from skimage import (color, io, segmentation)
from sklearn import (mixture, metrics, preprocessing, decomposition)
from scipy import (ndimage,io)
import glob, itertools
from ksptrack.cfgs import cfg
import logging
from sklearn.ensemble import RandomForestClassifier

class DataManager:

    def __init__(self,conf):
        self.conf = conf #Config Bunch object

        #Foreground/Background models (baggin)
        self.fg_marked = None #(frame, SP_label)
        self.bg_marked = None #(frame, SP_label)
        self.fg_pm_df = None #(frame, SP_label, proba)
        self.bg_pm_df = None #(frame, SP_label, proba)
        self.fg_marked_feats = None #(frame, SP_label, feature)
        self.bg_marked_feats = None #(frame, SP_label, feature)
        self.all_feats_df = None #(frame, SP_label, feature)

        self.logger = logging.getLogger('Dataset')

        self.labels = None
        self.flows = None
        self.labelContourMask = None #Superpixel contours
        self.centroids_loc = None #Superpixel centroids pandas frame
        self.loc_mat = None
        self.seen_feats_df = None #Descriptors of "seen" regions
        #self.sp_desc_df = None #Descriptors of all superpixels
        self.sp_entr_df = None #Costs of entrance to graph

        if(not os.path.exists(self.conf.precomp_desc_path)):
            self.logger.info('Feature directory does not exist... creating')
            os.mkdir(self.conf.precomp_desc_path)


    def relabel(self,save=False, who=[]):
        """ Relabel labels, sp_desc_df and sp_link_df to make labels contiguous
            who=['entr','desc','link','pm','centroids']
        """

        self.logger.info('Relabeling labels')

        has_changed = False
        if('desc' in who):
            sp_desc_df = self.get_sp_desc_from_file()
        if('link' in who):
            sp_link_df = self.get_link_data_from_file()

        with pbar(maxval=len(self.conf.frameFileNames)) as bar:
            for i in range(len(self.conf.frameFileNames)):
                bar.update(i)
                this_labels = self.get_labels()[..., i]
                sorted_labels = np.asarray(sorted(np.unique(this_labels).ravel()))
                if(np.any((sorted_labels[1:] - sorted_labels[0:-1])>1)):
                    has_changed = True
                    map_dict = {sorted_labels[i]:i for i in range(sorted_labels.shape[0])}
                    this_labels = utls.relabel(this_labels,map_dict)
                    self.labels[...,i] = this_labels
                    new_labels = np.asarray([i for i in range(sorted_labels.shape[0])])
                    if('centroids' in who):
                        self.centroids_loc.loc[(self.centroids_loc['frame'] == i),'sp_label'] = new_labels
                    if('entr' in who):
                        self.sp_entr_df.loc[(self.sp_entr_df['frame'] == i),'sp_label'] = new_labels
                    if('pm' in who):
                        self.fg_pm_df.loc[(self.fg_pm_df['frame'] == i),'sp_label'] = new_labels


                    if('desc' in who):
                        sp_desc_df.loc[(sp_desc_df['frame'] == i),'sp_label'] = new_labels

                    if('link' in who):
                        sp_link_df.loc[sp_link_df['input frame'] == i,'input label'].replace(map_dict, inplace=True)
                        sp_link_df.loc[sp_link_df['output frame'] == i,'output label'].replace(map_dict, inplace=True)

        #has_changed = True
        self.logger.info('done.')
        if(save & has_changed):
            self.logger.info('Saving to disk...')

            label_dict_out = dict()
            label_dict_out['sp_labels'] = self.labels
            mat_file_out = os.path.join(self.conf.dataInRoot,
                                        self.conf.dataSetDir,
                                        self.conf.frameDir,'sp_labels.mat')
            np_file_out = os.path.join(self.conf.dataInRoot,
                                       self.conf.dataSetDir,
                                       self.conf.frameDir,'sp_labels.npz')
            io.savemat(mat_file_out, label_dict_out)
            np.savez(np_file_out, **label_dict_out)

        elif(save & ~has_changed):
            self.logger.info('Nothing to change')

    def load_all_from_file(self):
        self.load_superpix_from_file()
        #self.load_seen_from_file()
        self.get_sp_desc_from_file()
        #self.load_spix_desc_from_file()
        #self.load_link_data_from_file()
        #self.load_entr_from_file()
        #self.load_pm_bg_from_file()
        self.load_pm_fg_from_file()
        #self.load_pm_all_feats()

    def get_flows(self):
        file_ = os.path.join(self.conf.precomp_desc_path, 'flows.npz')
        if(not os.path.exists(file_)):
            self.flows_mat_to_np()
        self.flows = dict()
        self.logger.info('Loading optical flows...')
        npzfile = np.load(file_)
        self.flows['bvx'] = npzfile['bvx']
        self.flows['fvx'] = npzfile['fvx']
        self.flows['bvy'] = npzfile['bvy']
        self.flows['fvy'] = npzfile['fvy']
        return self.flows

    def get_labels(self):
        return self.load_labels_if_not_exist()

    def load_centroids_if_not_exist(self):

        if(self.labels is None):

            self.labels = np.load(os.path.join(self.conf.dataInRoot,                                      self.conf.dataSetDir,                                           self.conf.frameDir,'sp_labels.npz'))['sp_labels']

        return self.labels

    def load_labels_contours_if_not_exist(self):

        if(self.labelContourMask is None):
            self.labelContourMask = np.load(os.path.join(self.conf.precomp_desc_path,'sp_labels_tsp_contours.npz'))['labelContourMask'].transpose((1,2,0))

    def load_labels_if_not_exist(self):

        if(self.labels is None):

            self.labels = np.load(os.path.join(self.conf.precomp_desc_path,
                                               'sp_labels.npz'))['sp_labels']

            #mat_file_out = os.path.join(self.conf.dataInRoot,
            #                            self.conf.dataSetDir,
            #                            'EE',
            #                            'sp_labels_ml.mat')
            #self.labels = scipy.io.loadmat(mat_file_out)['labels']

        return self.labels

    @property
    def sp_desc_df(self):
        return self.get_sp_desc_from_file()

    @property
    def sp_desc_means(self):
        return self.get_sp_desc_means_from_file()


    def load_superpix_from_file(self):

        self.logger.info('loading superpixel data (labels,centroids)')
        self.centroids_loc = pd.read_pickle(
            os.path.join(self.conf.precomp_desc_path, 'centroids_loc_df.p'))
        self.load_labels_if_not_exist()

    def get_sp_desc_from_file(self):

        if(self.conf.feats_graph == 'overfeat'):
            fname = 'overfeat.p'
        elif(self.conf.feats_graph == 'unet'):
            fname = 'sp_desc_unet_rec.p'
        elif(self.conf.feats_graph == 'unet_gaze'):
            gaze_fname = os.path.splitext(self.conf.csvFileName_fg)[0]
            gaze_num = gaze_fname[-1]
            #fname = 'sp_desc_ung_g' + gaze_num + '.p'
            fname = 'sp_desc_ung.p'
        elif(self.conf.feats_graph == 'unet_gaze_cov'):
            gaze_fname = os.path.splitext(self.conf.csvFileName_fg)[0]
            fname = 'sp_desc_ung_' + gaze_fname + '.p'
        elif(self.conf.feats_graph == 'scp'):
            fname = 'sp_desc_df.p'
        elif(self.conf.feats_graph == 'hsv'):
            fname = 'sp_desc_hsv_df.p'
        elif(self.conf.feats_graph == 'vgg16'):
            fname = 'sp_desc_vgg16.p'

        path = os.path.join(self.conf.precomp_desc_path, fname)
        if(os.path.exists(path)):
            out =  pd.read_pickle(path)

            self.logger.info("Loaded features: " + path)
        else:
            self.logger.info("Couldnt find features: {}".format(path))
            return None

        return out

    def get_sp_desc_means_from_file(self):

        path = os.path.join(self.conf.precomp_desc_path, 'sp_desc_means.p')
        if(os.path.exists(path)):
            out =  pd.read_pickle(os.path.join(path))
        else:
            return None

        return out

    def load_sp_desc_means_from_file(self):
        self.logger.info("Loading sp means ")
        self.sp_desc_means = pd.read_pickle(os.path.join(self.conf.precomp_desc_path, 'sp_desc_means.p'))

    def load_pm_all_feats(self):
        self.logger.info("Loading PM all feats")
        self.all_feats_df = pd.read_pickle(os.path.join(self.conf.precomp_desc_path, 'all_feats_df.p'))

    def load_pm_fg_from_file(self):
        self.logger.info("Loading PM foreground")
        self.fg_marked = np.load(os.path.join(self.conf.precomp_desc_path, 'fg_marked.npz'))['fg_marked']
        self.fg_pm_df = pd.read_pickle(
            os.path.join(self.conf.precomp_desc_path, 'fg_pm_df.p'))
        self.fg_marked_feats = np.load(os.path.join(self.conf.precomp_desc_path, 'fg_marked_feats.npz'))['fg_marked_feats']

    def load_pm_bg_from_file(self):
        self.logger.info("Loading PM background")
        self.bg_marked = np.load(os.path.join(self.conf.precomp_desc_path, 'bg_marked.npz'))['bg_marked']

        self.bg_pm_df = pd.read_pickle(
            os.path.join(self.conf.precomp_desc_path, 'bg_pm_df.p'))
        self.bg_marked_feats = np.load(os.path.join(self.conf.precomp_desc_path, 'bg_marked_feats.npz'))['bg_marked_feats']


    def calc_oflow(self, save=True):

        save_path = os.path.join(self.conf.precomp_desc_path)
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)

        oflow_extractor = oflowx.OpticalFlowExtractor(save_path,
                                                      self.conf.oflow_alpha,
                                                      self.conf.oflow_ratio,
                                                      self.conf.oflow_minWidth,
                                        self.conf.oflow_nOuterFPIterations,
                                        self.conf.oflow_nInnerFPIterations,
                                        self.conf.oflow_nSORIterations)
        oflow_extractor.extract(self.conf.frameFileNames,
                                save_path)

    def calc_superpix(self,save=True):
        """
        Makes centroids and contours
        """

        self.labelContourMask = list()
        spix_extr = svx.SuperpixelExtractor()
        self.labels = spix_extr.extract(self.conf.frameFileNames,
                          os.path.join(self.conf.precomp_desc_path,
                                       'sp_labels.npz'),
                          self.conf.compactness,
                          self.conf.reqdsupervoxelsize,
                          save)

        self.logger.info("Loading npz label map")

        labelContourMask = list()
        self.logger.info("Generating label contour maps")

        for im in range(self.labels.shape[2]):
            # labels values are not always "incremental" (values are skipped).
            labelContourMask.append(
                segmentation.find_boundaries(self.labels[:, :, im]))
            #labels[:,:,im] = fixedLabels
        self.logger.info("Saving label contour maps")
        data = dict()
        data['labelContourMask'] = labelContourMask
        np.savez(
            os.path.join(self.conf.precomp_desc_path,
                        'sp_labels_tsp_contours.npz'), **data)
        self.labelContourMask = np.load(
            os.path.join(
                self.conf.precomp_desc_path,
                'sp_labels_tsp_contours.npz'))['labelContourMask']

        #self.labels = self.labels[:, :, np.arange(self.conf.seqStart, self.conf.seqEnd+1)]
        self.labelContourMask = np.asarray(self.labelContourMask)
        #labelContourMask = np.transpose(labelContourMask,(1,2,0))

        self.logger.info('Getting centroids...')
        self.centroids_loc = spix.getLabelCentroids(self.labels)

        if(save):

            self.centroids_loc.to_pickle(
                os.path.join(self.conf.dataInRoot, self.conf.dataSetDir, self.conf.precomp_desc_path, 'centroids_loc_df.p'))

        return self.labels, self.labelContourMask
                # End superpixelization--------------------------------------------------

    def calc_bagging(self,
                     marked_arr,
                     marked_feats=None,
                     all_feats_df=None,
                     mode='foreground',
                     feat_fields = ['desc'],
                     T = 100,
                     max_n_feats = 0.25,
                     max_depth = 5,
                     max_samples=2000,
                     n_jobs = 1):
        """
        Computes "Bagging" transductive probabilities using marked_arr as positives.
        """

        if(n_jobs > 1):
            this_marked_feats,  this_pm_df = bag.calc_bagging_para(
                T,
                max_depth,
                max_n_feats,
                marked_arr,
                all_feats_df=all_feats_df,
                feat_fields=['desc'],
                max_samples=max_samples,
                n_jobs=n_jobs)
        else:
            this_marked_feats,  this_pm_df = bag.calc_bagging(
                        T,
                        max_depth,
                        max_n_feats,
                        marked_arr,
                        all_feats_df=all_feats_df,
                        feat_fields=['desc'],
                        max_samples=max_samples)

        if(mode == 'foreground'):
            self.fg_marked_feats = this_marked_feats
            self.fg_pm_df = this_pm_df
        else:
            self.bg_marked_feats = this_marked_feats
            self.bg_pm_df = this_pm_df

        return self.fg_pm_df



    def __calc_sp_feats_unet(self,
                             model,
                             feats_name,
                             desc_name,
                             locs2d=None,
                             save_dir=None):
        """ Computes UNet features in Autoencoder-mode
                Train/forward-propagate Unet and save features/weights
                """

        self.logger.info('Calculating U-Net features.')

        #img_all_dirs = cfg.seq_type_dict[self.conf.seq_type]

        all_imgs = self.conf.frameFileNames


        # get 2d locations as numpy array
        if(self.conf.csvFileType == 'anna'):
            self.locs2d_arr = locs2d[:,3:]
        elif(self.conf.csvFileType == 'pandas'):
            self.locs2d_arr = locs2d.as_matrix()[:,4:]

        # get a sample image to set correct input size of network
        sample_img_path = all_imgs[0]

        #weight_dir_existed = False
        if not os.path.exists(save_dir):
            self.logger.info('U-Net output directory: ' + str(save_dir) + ' does not exist...creating.')
            os.makedirs(save_dir)
        else:
            self.logger.info('Found directory: ' + str(save_dir))

        # constructor (specify "img_height" and "img_width" for a specific input size)
        unet_mod = model(self.conf, save_dir, sample_img_path, img_height=None, img_width=None, locs2d=self.locs2d_arr)
        weights_fname = unet_mod.get_weights_path()
        feat_fname = os.path.join(save_dir, feats_name)
        sp_desc_fname = os.path.join(save_dir, desc_name)

        if not os.path.exists(weights_fname):

            self.logger.info('Weight file: ' + str(weights_fname) + ' does not exist...train network.')

            # train the model on all image sets
            unet_mod.train(self.conf, all_imgs, locs2d)
        else:
            self.logger.info('Weight file: ' + str(weights_fname) + ' exists.')

        # always do forward propagation, since we do not know if the existing feature file would be created with the
        # latest model weights
        # note: forward propagation only on the individual set images
        if not os.path.exists(feat_fname):
            self.logger.info('Forward propagating...')
            unet_mod.forward_prop(self.conf, feat_fname, all_imgs)
        else:
            self.logger.info('Downscaled features already exist in')
            self.logger.info(feat_fname)

        if not os.path.exists(sp_desc_fname):
            self.logger.info('Generating features on superpixels (interpolating)')
            self.logger.info('Number of jobs: ' + str(self.conf.unet_interp_n_jobs))
            sp_desc_df = unet_mod.interp_features(self.conf, feat_fname, self.get_labels(),
                                                n_jobs=self.conf.unet_interp_n_jobs)
            if (save_dir is not None):
                self.logger.info('Saving sp descriptors to: ' + sp_desc_fname)
                sp_desc_df.to_pickle(sp_desc_fname)
        else:
            self.logger.info('sp descriptors : ' + sp_desc_fname + ' already exist.')


    def calc_sp_feats_vgg16(self, save_dir=None):
        """ Computes VGG16 features
         and save features
        """
        from ksptrack.feat_extractor.myvgg16 import MyVGG16
        from ksptrack.feat_extractor.feat_data_loader import PatchDataLoader

        fnames = self.conf.frameFileNames
        sp_labels = self.load_labels_if_not_exist()

        myvgg16 = MyVGG16(cuda=self.conf.vgg16_cuda)

        with pbar(maxval=len(fnames)) as bar:
            for f_ind, f in enumerate(fnames):
                bar.update(f_ind)
                feats = list()
                patch_loader = PatchDataLoader(f,
                                               sp_labels[..., f_ind],
                                               self.conf.vgg16_size,
                                               myvgg16.transform,
                                               batch_size=\
                                               self.conf.vgg16_batch_size)

                im_feats_save_path = os.path.join(self.conf.precomp_desc_path,
                                                  'im_feat_{}.p'.format(f_ind))

                if(not os.path.exists(im_feats_save_path)):
                    for patches, labels in patch_loader.data_loader:
                        feats_ = myvgg16.get_features(patches)
                        feats += [(f_ind, l, f_)
                                for (l, f_) in zip(np.asarray(labels),
                                                    np.asarray(feats_))]

                    feats = pd.DataFrame(
                        feats, columns=["frame", "sp_label", "desc"])
                    feats.sort_values(['frame', 'sp_label'], inplace=True)
                    feats.to_pickle(im_feats_save_path)

        # Make single feature file for convenience
        all_feats_path = os.path.join(self.conf.precomp_desc_path,
                                        'sp_desc_vgg16.p')

        if(not os.path.exists(all_feats_path)):
            im_feats_paths = sorted(glob.glob(
                os.path.join(self.conf.precomp_desc_path,
                            'im_feat_*.p')))
            all_feats = [pd.read_pickle(p_) for p_ in im_feats_paths]

            feats_df = pd.concat(all_feats)
            feats_df.sort_values(['frame', 'sp_label'], inplace=True)

            self.logger.info("Saving all features at {}".format(all_feats_path))
            feats_df = feats_df.reset_index()
            feats_df.to_pickle(all_feats_path)
        else:
            all_feats = pd.read_pickle(all_feats_path)

        return all_feats


    def calc_sp_feats_unet_rec(self, save_dir=None):
        """ Computes UNet features in Autoencoder-mode
        Train/forward-propagate Unet and save features/weights
        """

        from unet.UNetRec import UNetRec
        self.__calc_sp_feats_unet(UNetRec,
                                  'feats_unet_rec.h5',
                                  'sp_desc_unet_rec.p',
                                  save_dir=save_dir)


    def calc_sp_feats_unet_gaze_rec(self, locs2d, save_dir=None):
        """ Computes UNet features in Autoencoder-mode
        Train/forward-propagate Unet and save features/weights
        """

        h5_fname = 'feat_ung.h5'
        df_fname = 'sp_desc_ung.p'

        from ksptrack.nets.UNetGazeRec import UNetGazeRec
        self.__calc_sp_feats_unet(UNetGazeRec,
                                  h5_fname,
                                  df_fname,
                                  locs2d=locs2d,
                                  save_dir=save_dir)

    def calc_entrance(self,save=False):
        """
        Histogram intersections of seen superpixels with others (unused)
        """

        self.load_seen_if_not_exist()
        labels = self.load_labels_if_not_exist()
        sp_desc_df = self.get_sp_desc_from_file()

        im_shape = labels[0,...].shape

        self.logger.info("Calculating entrance (seen-to-sp) histogram intersections")
        sp_entr = []
        seen_feats_mat = self.seen_feats_df.as_matrix()
        with pbar(maxval=sp_desc_df.shape[0]) as bar:
            for i in range(sp_desc_df.shape[0]):
                bar.update(i)
                this_sp_inters = []
                this_sp_color_dists = []
                this_desc = sp_desc_df['desc'][sp_desc_df.index[i]]
                for j in range(self.seen_feats_df.shape[0]):
                    this_frame = sp_desc_df['frame'][sp_desc_df.index[i]]
                    this_label = sp_desc_df['sp_label'][sp_desc_df.index[i]]
                    this_inter = utls.hist_inter(this_desc, self.seen_feats_df['scp'][j])
                    this_sp_inters.append(this_inter)
                sp_entr.append((this_frame, this_label, this_sp_inters))

        self.sp_entr_df = pd.DataFrame(
            sp_entr, columns=['frame', 'sp_labels', 'inters'])

        #Sort by frame, sp_label
        self.sp_entr_df.sort_values(['frame','sp_labels'],inplace=True)


        if(save):
            self.sp_entr_df.to_pickle(
                os.path.join(self.conf.precomp_desc_path, 'sp_entr_df.p'))

    def calc_linking(self, save=False):
        """ Computes linking costs of neighboring superpixels
        """

        self.load_superpix_from_file()
        sp_desc_df = self.get_sp_desc_from_file()

        self.logger.info("Calculating linking histogram intersections")
        sp_link = []
        self.centroids_loc = pd.read_pickle(
            os.path.join(self.conf.dataInRoot, self.conf.dataSetDir, self.conf.precomp_desc_path,
                        'centroids_loc_df.p'))
        self.loc_mat = self.centroids_loc.as_matrix()
        self.sp_desc_mat = sp_desc_df.as_matrix()
        with pbar(
                maxval=np.unique(sp_desc_df['frame'])[0:-1].shape[0]) as bar:
            for i in np.unique(sp_desc_df['frame'])[0:-1]:
                bar.update(i)
                this_frame_labels = np.where(sp_desc_df['frame'] == i)[0]
                next_frame_labels = np.where(sp_desc_df['frame'] == i + 1)[0]
                for j in itertools.product(this_frame_labels,
                                        next_frame_labels):
                    idx_in = j[0]
                    idx_out = j[1]
                    loc1 = self.loc_mat[idx_in, :][2:]
                    loc2 = self.loc_mat[idx_out, :][2:]
                    this_loc_dist = np.linalg.norm(loc1 - loc2)

                    if (this_loc_dist < .18):

                        sp_link.append([
                            i, sp_desc_df['sp_label'][sp_desc_df.index[idx_in]],
                            i + 1,
                            sp_desc_df['sp_label'][sp_desc_df.index[idx_out]],
                            0.5, loc1, loc2, this_loc_dist
                        ])

        self.sp_link_df = pd.DataFrame(
            sp_link,
            columns=[
                'input frame', 'input label', 'output frame', 'output label',
                'hist_inter', 'loc_in', 'loc_out', 'loc_dist'
            ])
        if(save):
            self.sp_link_df.to_pickle(
                os.path.join(self.conf.dataInRoot, self.conf.dataSetDir, self.conf.precomp_desc_path,
                            'sp_link_df.p'))

    def load_seen_if_not_exist(self):

        if(self.seen_feats_df is None):
            self.load_seen_desc_from_file()


    def get_pm_array(self,mode='foreground',save=False,frames=None):
        """ Returns array same size as labels with probabilities of bagging model
        """

        self.load_labels_if_not_exist()
        scores = self.labels.copy().astype(float)
        if(mode == 'foreground'):
            pm_df = self.fg_pm_df
        else:
            pm_df = self.bg_pm_df

        if(frames is None): #Make all frames
            frames = np.arange(scores.shape[-1])
        else:
            frames = np.asarray(frames)

        i = 0
        self.logger.info('Generating PM array')
        with pbar(
                maxval=frames.shape[0]) as bar:
            for f in frames:
                bar.update(i)
                this_frame_pm_df = pm_df[pm_df['frame'] == f]
                dict_keys = this_frame_pm_df['sp_label']
                dict_vals = this_frame_pm_df['proba']
                dict_map = dict(zip(dict_keys, dict_vals))
                for k, v in dict_map.items():
                    scores[scores[...,f]==k,f] = v
                i += 1

        if(save):

            if(mode == 'foreground'):
                data = dict()
                data['pm_scores'] = scores
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'pm_scores_fg_orig.npz'), **data)
            else:
                data = dict()
                data['pm_scores'] = scores
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'pm_scores_fg_orig.npz'), **data)

        return scores

    def update_marked_sp(self,marked,mode='foreground'):
        """ Updates marked SP for bagging (with ksp SPs)
        """

        if(mode == 'foreground'):
            old_marked = self.fg_marked
        else:
            old_marked = self.bg_marked

        #Insert sp in marked if it doesn't exist already
        if(old_marked is not None):
            for i in range(marked.shape[0]):
                match_rows_frames = np.where(old_marked[:,0] == marked[i,0])
                match_rows = np.where(old_marked[match_rows_frames,1].ravel() == marked[i,1])
                if(match_rows[0].size == 0): #No match found, current marked SP is added
                    old_marked = np.insert(old_marked,0,marked[i,...],axis=0)


            idx_sort = np.argsort(old_marked[:,0])
            old_marked = old_marked[idx_sort,:]

            if(mode == 'foreground'):
                self.fg_marked = old_marked
            else:
                self.bg_marked = old_marked
        else:
            if(mode == 'foreground'):
                self.fg_marked = marked
            else:
                self.bg_marked = marked

    def calc_pm(self,coords_arr,save=False,marked_feats=None,
                all_feats_df=None,in_type='csv_normalized',
                mode='foreground',
                feat_fields=['desc'],
                T = 100,
                max_n_feats = 0.25,
                max_depth = 5,
                max_samples = 2000,
                n_jobs = 1):
        """ Main function that extracts or updates gaze coordinates and computes transductive learning model (bagging)
            Inputs:
                save: Boolean (save to file?)
                in_type: {'csv_normalized,','centroids'}
                mode: {'foreground','background'}
        """


        self.load_labels_if_not_exist()

        self.logger.info('--- Generating probability map foreground')
        self.logger.info("T: " + str(self.conf.T))
        self.logger.info("n_bins: " + str(self.conf.n_bins))

        # Convert input to marked (if necessary). This is used only once (from csv gaze file)
        if(in_type == 'csv_normalized'):
            marked_arr = np.empty((coords_arr.shape[0],2))
            for i in range(coords_arr.shape[0]):
                ci, cj = csv.coord2Pixel(coords_arr[i, 3],
                                         coords_arr[i, 4],
                                         self.labels.shape[1],
                                         self.labels.shape[0])
                marked_arr[i,...] = (coords_arr[i, 0],
                                     self.labels[ci,cj,int(coords_arr[i, 0])])
        #elif(in_type == 'pandas'):
        #    marked_arr = list()
        #    for f in coords_arr['frame']:
        #        ci, cj = gaze.gazeCoord2Pixel(coords_arr.loc[coords_arr['frame'] == f, 'x'], coords_arr.loc[coords_arr['frame'] == f, 'y'],
        #                                    self.labels.shape[1], self.labels.shape[0])
        #        marked_arr.append((f,self.labels[ci,cj,f]))
        #    marked_arr = np.asarray(marked_arr)
        else:
            marked_arr = coords_arr

        if(mode == 'foreground'):
            self.fg_marked = marked_arr #Set first marked sp array

        self.calc_bagging(marked_arr,
                          marked_feats=marked_feats,
                          all_feats_df=all_feats_df,
                          mode=mode,
                          feat_fields=feat_fields,
                          T=self.conf.T,
                          max_n_feats=max_n_feats,
                          max_depth=max_depth,
                          max_samples = max_samples,
                          n_jobs=n_jobs)

        if(save):

            if(mode == 'foreground'):
                data = dict()
                data['fg_marked'] = self.fg_marked
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'fg_marked.npz'), **data)
                self.fg_pm_df.to_pickle(os.path.join(self.conf.precomp_desc_path,
                                                     'fg_pm_df.p'))
                data = dict()
                data['fg_marked_feats'] = self.fg_marked_feats
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'fg_marked_feats.npz'), **data)
            else:
                data = dict()
                data['bg_marked'] = self.bg_marked
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'bg_marked.npz'), **data)
                self.bg_pm_df.to_pickle(os.path.join(self.conf.precomp_desc_path,
                                                     'bg_pm_df.p'))
                data = dict()
                data['bg_marked_feats'] = self.bg_marked_feats
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'bg_marked_feats.npz'), **data)

    def calc_pm_rf(self,coords_arr,save=False,marked_feats=None,
                all_feats_df=None,in_type='csv_normalized',
                mode='foreground',
                feat_fields=['desc']):
        """ Main function that extracts or updates gaze coordinates and computes transductive learning model (bagging)
            Inputs:
                save: Boolean (save to file?)
                in_type: {'csv_normalized,','centroids'}
                mode: {'foreground','background'}
        """


        self.load_labels_if_not_exist()

        self.logger.info('--- Generating probability map foreground')
        self.logger.info("T: " + str(self.conf.T))
        self.logger.info("n_bins: " + str(self.conf.n_bins))

        #Convert input to marked (if necessary). This is used only once (from csv gaze file)
        marked_arr = np.empty((coords_arr.shape[0],2))
        if(in_type == 'csv_normalized'):
            for i in range(coords_arr.shape[0]):
                ci, cj = gaze.gazeCoord2Pixel(coords_arr[i, 3], coords_arr[i, 4],
                                            self.labels.shape[1], self.labels.shape[0])
                marked_arr[i,...] = (i,self.labels[ci,cj,i])
            if(mode == 'foreground'):
                self.fg_marked = marked_arr #Set first marked sp array
            else:
                self.bg_marked = marked_arr #Set first marked sp array
        else:
            marked_arr = coords_arr


        X_neg,X_pos =  bag.make_samples(marked_arr,
                                        marked_feats=None, all_feats_df=all_feats_df,
                                        mode='foreground',
                                        feat_fields=['desc'],
                                        remove_marked=False)

        y = np.concatenate((np.ones(X_pos.shape[0]),np.zeros(X_neg.shape[0])))
        X = np.concatenate((X_pos[:,2:],X_neg[:,2:]))

        clf = RandomForestClassifier(n_estimators=self.conf.T,
                                     verbose=1,
                                     max_depth=self.conf.max_depth,
                                     n_jobs=4)
        clf.fit(X, y)
        clf_probs = clf.predict_proba(X)

        data_frames = np.vstack((X_pos[:,0].reshape(-1,1), X_neg[:,0].reshape(-1,1))).astype(int)
        data_labels = np.vstack((X_pos[:,1].reshape(-1,1), X_neg[:,1].reshape(-1,1))).astype(int)
        data_probas = clf_probs[:,1].reshape(-1,1)
        pm_df = pd.DataFrame({'frame':data_frames.ravel(),'sp_label':data_labels.ravel(),'proba':data_probas.ravel()})

        pm_df.sort_values(['frame','sp_label'],inplace=True)
        self.fg_pm_df = pm_df

        if(save):

            if(mode == 'foreground'):
                data = dict()
                data['fg_marked'] = self.fg_marked
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'fg_marked.npz'), **data)
                self.fg_pm_df.to_pickle(os.path.join(self.conf.precomp_desc_path,
                                                     'fg_pm_df.p'))
                data = dict()
                data['fg_marked_feats'] = self.fg_marked_feats
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'fg_marked_feats.npz'), **data)
            else:
                data = dict()
                data['bg_marked'] = self.bg_marked
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'bg_marked.npz'), **data)
                self.bg_pm_df.to_pickle(os.path.join(self.conf.precomp_desc_path,
                                                     'bg_pm_df.p'))
                data = dict()
                data['bg_marked_feats'] = self.bg_marked_feats
                np.savez(os.path.join(self.conf.precomp_desc_path,
                                      'bg_marked_feats.npz'), **data)
