import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
import tqdm
from ksptrack.utils import superpixel_utils as sputls
from ksptrack.utils.base_dataset import BaseDataset
from skimage import io, segmentation


class SuperpixelExtractor:
    def __init__(self,
                 root_path,
                 desc_dir='precomp_desc',
                 compactness=20.,
                 n_segments=1200):
        self.desc_path = pjoin(root_path, desc_dir)
        self.desc_dir = desc_dir
        self.root_path = root_path

        self.labels_ = None
        self.labels_contours_ = None
        self.centroids_loc_ = None
        self.sp_desc_df_ = None

        if (not os.path.exists(self.desc_path)):
            print(self.desc_path, ' directory does not exist... creating')
            os.mkdir(self.desc_path)

        self.compactness = compactness
        self.n_segments = n_segments

    @property
    def labels(self):

        if (self.labels_ is None):
            self.labels_ = np.load(
                os.path.join(self.desc_path, 'sp_labels.npy'))
        return self.labels_

    @property
    def labels_contours(self):

        if (self.labels_contours_ is None):
            self.labels_contours_ = np.load(
                os.path.join(self.desc_path, 'sp_labels_tsp_contours.npz')
            )['labels_contours'].transpose((1, 2, 0))
        return self.labels_contours_

    @property
    def centroids_loc(self):

        if (self.centroids_loc_ is None):
            centroid_path = os.path.join(self.desc_path, 'centroids_loc_df.p')
            print('loading superpixel centroids: {}'.format(centroid_path))
            self.centroids_loc_ = pd.read_pickle(centroid_path)
        return self.centroids_loc_

    def run(self, do_save=True):
        """
        Makes centroids and contours
        """

        dset = BaseDataset(self.root_path)
        if (not os.path.exists(pjoin(self.desc_path, 'sp_labels.npy'))):

            print('Running SLIC on {} images with {} labels'.format(
                len(dset), self.n_segments))
            labels = np.array([
                segmentation.slic(s['image'],
                                  n_segments=self.n_segments,
                                  compactness=self.compactness,
                                  start_label=0) for s in dset
            ]).astype(np.uint16)
            print('Saving labels to {}'.format(self.desc_path))
            np.save(os.path.join(self.desc_path, 'sp_labels.npy'), labels)

        if (not os.path.exists(pjoin(self.desc_path,
                                     'sp_labels_contours.npz'))):
            self.labels_contours_ = list()
            print("Generating label contour maps")

            for l in self.labels:
                # labels values are not always "incremental" (values are skipped).
                self.labels_contours_.append(
                    segmentation.find_boundaries(l).astype(bool))

            self.labels_contours_ = np.array(self.labels_contours_)
            print("Saving labels")
            data = dict()
            data['labels_contours'] = self.labels_contours
            np.savez(os.path.join(self.desc_path, 'sp_labels_contours.npz'),
                     **data)

        if (not os.path.exists(pjoin(self.desc_path, 'centroids_loc_df.p'))):
            print('Getting centroids...')
            self.centroids_loc_ = sputls.getLabelCentroids(self.labels)

            self.centroids_loc_.to_pickle(
                os.path.join(self.desc_path, 'centroids_loc_df.p'))

        if (do_save and
                not os.path.exists(pjoin(self.desc_path, 'spix_previews'))):
            print('Saving slic previews to {}'.format(
                pjoin(self.desc_path, 'spix_previews')))
            previews_dir = os.path.join(self.desc_path, 'spix_previews')
            if (not os.path.exists(previews_dir)):
                os.makedirs(previews_dir)
            for i, sample in enumerate(dset):
                fname = os.path.join(previews_dir,
                                     'frame_{0:04d}.png'.format(i))

                im = sputls.drawLabelContourMask(sample['image'],
                                                 self.labels[i, ...])
                io.imsave(fname, im)
