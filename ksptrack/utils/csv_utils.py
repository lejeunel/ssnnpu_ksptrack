import getopt
import csv
import numpy as np
import os.path
from skimage.draw import disk
from scipy import ndimage
import pandas as pd


def draw2DPoint(locs2d, frame_num, img, radius=2, color=(0, 255, 0)):
    if (frame_num in locs2d[:, 0]):
        height, width, _ = img.shape
        frame_row = np.where(locs2d[:, 0] == frame_num)[0][0]
        i, j = coord2Pixel(locs2d[frame_row, 3], locs2d[frame_row, 4], width,
                           height)
        rr, cc = disk((i, j), radius, shape=(height, width))
        img[rr, cc, 0] = color[0]
        img[rr, cc, 1] = color[1]
        img[rr, cc, 2] = color[2]

    return img


def isKeyPressed(arr, frameNum):
    return bool(arr[frameNum, 2])


def pix2Norm(j, i, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    x = j / (width - 1)
    y = i / (height - 1)

    return x, y


def coord2Pixel(x, y, width, height, round_to_int=True):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = x * (width - 1)
    i = y * (height - 1)

    if (round_to_int):
        j = int(np.round(j, 0))
        i = int(np.round(i, 0))

    return i, j


def pixCoord2SPcentroid(coords_arr, labels):

    centroids_out = np.empty(coords_arr.shape)
    for i in range(coords_arr.shape[0]):
        this_centroid = ndimage.measurements.center_of_mass(
            labels[i, :, :] == labels[i, coords_arr[i, 4], coords_arr[i, 3]])
        centroids_out[i, ...] = (coords_arr[i, 0:3], this_centroid[1],
                                 this_centroid[0])

    return centroids_out


def coord2normalized(x, y, width, height):
    """
    Returns i and j (line/column) coordinate of gaze point given image dimensions
    """

    normFactor = np.linalg.norm((width - 1, height - 1))

    x = x * (width - 1) / normFactor
    y = y * (height - 1) / normFactor

    return x, y


def readCsv(filePath, as_pandas=False):

    if (as_pandas):
        arr = pd.read_csv(filePath, skiprows=4, delimiter=";")
    else:
        arr = np.loadtxt(open(filePath, "rb"), delimiter=";", skiprows=5)

    return arr


#def writeCsv(a, path, field_names = ['']):
#    with open(path, newline='') as csvfile:
#        reader = csv.DictReader(csvfile)
#        for row in reader:
#            print(row['first_name'], row['last_name'])
