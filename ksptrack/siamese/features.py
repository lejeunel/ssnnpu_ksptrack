import collections
import math
import numpy as np
import scipy.ndimage.filters
from skimage import segmentation, morphology, transform
import matplotlib.pyplot as plt


class Features:
    def __init__(self,
                 feature_maps,
                 labels,
                 n_region,
                 ball_radius=5):
        self.feature_maps = feature_maps
        self.labels = labels
        self.label_contours = segmentation.find_boundaries(labels)
        self.selem = morphology.disk(ball_radius)
        self.selem_boundary = morphology.disk(1)

    def similarity(self, i, j):
        # find border of two neighboring regions
        mask = morphology.binary_dilation(self.labels == i,
                                          self.selem_boundary)
        mask *= morphology.binary_dilation(self.labels == j,
                                           self.selem_boundary)
        mask = morphology.binary_dilation(mask, self.selem)

        sim = np.median([a[mask] for a in self.boundary_maps])

        return sim


    def merge(self, i, j):
        new_region_id = len(self.size)
        return new_region_id
