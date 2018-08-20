import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io
from skimage import color
import sys
import os
import progressbar
from . import csv_utils as csv

def getLabelCentroids(labels):
    """
    labels is a list of positive-integer matrix whose values refer to a label. Values must be contiguous.
    Returns: List of same length as labels, containing arrays with first element containing value of label, and second and third element containing the x and y coordinates of centroid.
    """

    nFrames = labels.shape[2]
    centroids = []

    #normFactor = np.linalg.norm((labels.shape[0],labels.shape[1]))

    centroid_list = []
    with progressbar.ProgressBar(maxval=nFrames) as bar:
        for i in range(nFrames):
            bar.update(i)
            idxLabels = np.unique(labels[:,:,i])
            for j in range(len(idxLabels)):
                thisMask = (labels[..., i] == idxLabels[j])
                pos = np.asarray(ndimage.measurements.center_of_mass(thisMask))
                pos_norm = csv.pix2Norm(pos[1],
                                            pos[0],
                                            labels.shape[1],
                                            labels.shape[0])
                centroid_list.append([i,
                                      int(idxLabels[j]),
                                      pos_norm[0],
                                      pos_norm[1]])
    centroids = pd.DataFrame(centroid_list,
                             columns=['frame',
                                      'sp_label',
                                      'pos_norm_x',
                                      'pos_norm_y'])

    return(centroids)

def drawLabelContourMask(img,
                         label):

    red_img = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    mask_inv = np.invert(label)
    i_mask,j_mask = np.where(label)
    i_mask_inv,j_mask_inv = np.where(mask_inv)
    result = img.copy()
    result[i_mask,j_mask,:] = 0
    result[i_mask,j_mask,0] = 255

    return result
