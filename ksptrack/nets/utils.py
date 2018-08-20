import numpy as np
import os
import random as rd
from scipy.interpolate import interp2d
import scipy as sp
from keras import backend as K
import keras
from progressbar import Bar, Percentage, ProgressBar
import scipy.misc as scm
import matplotlib.pyplot as plt

class EvaluateCallback(keras.callbacks.Callback):
    """
    This callback class passes an image through the network during training, and saves the
    network output (save_every defines the number of batches after which a new prediction is made).
    It serves as a monitor to see how a network slowly learns to identify the interesting structures.
    """
    def __init__(self,
                 original,
                 truth,
                 out_path=None,
                 verbose=True,
                 save_every=50):
        keras.callbacks.Callback.__init__(self)
        self.n_batch = 0
        self.n_epoch = 0
        self.out_path = out_path
        self.save_every = save_every
        self.verbose = verbose
        self.image = original
        self.truth = truth

        # create path to save intermediate results, if it does not exist
        os.makedirs(out_path, exist_ok=True)

        # 'th': image is of shape (nrows, ncols, nChannels)
        if K.image_dim_ordering() == "th":
            original = np.swapaxes(np.swapaxes(original, 0, 1), 1, 2)

        if out_path is not None:
            #Make 3 channels if necessary
            if(original.shape[-1] == 1):
                orig_out = np.tile(original,(1,1,3))
            else:
                orig_out = original
            if(truth.shape[-1] == 1):
                truth_out = np.tile(truth,(1,1,3))
            else:
                truth_out = truth
            sp.misc.imsave(
                os.path.join(self.out_path, "original.png"),
                orig_out)
            sp.misc.imsave(
                os.path.join(self.out_path, "truth.png"), truth_out)

    def on_batch_end(self, batch, logs={}):
        if self.n_batch % self.save_every == 0:
            if self.verbose:
                loss = self.model.evaluate(self.image, self.truth)
                print("Current validation loss: ", loss)
                print()

            if self.out_path is not None:
                mask = self.model.predict(self.image[np.newaxis,...], batch_size=1)[0]

                if self.verbose:
                    with open(
                            os.path.join(self.out_path, "loss.txt"),
                            mode="w") as f:
                        f.write(str(loss))
                        f.flush()
                sp.misc.imsave(
                    os.path.join(
                        self.out_path,
                        "{0}x{1}_mask.png".format(self.n_epoch, self.n_batch)),
                    np.squeeze(mask))
        self.n_batch += 1

    def on_epoch_end(self, batch, lgos={}):
        self.n_epoch += 1

def shuffle_forward(arr, seed=None, order=None):
    """
    Shuffle an arbitrary nparray in the first dimension

    :param arr:  the array
    :param seed: the seed value
    :return:     (tuple) \n
                 shuffled array \n
                 order needed for reshuffeling
    """
    if order is None:
        order = np.arange(len(arr))
        rd.seed(seed)
        rd.shuffle(order)
    return np.array(arr)[order, ...], order


def shuffle_backward(arr, order):
    """
    Shuffle back an array by its order

    :param arr:   the array
    :param order: the generated order by "shuffle_forward()"
    :return:      the reshuffled array
    """
    arr_out = np.array(arr)
    for i, j in enumerate(order):
        arr_out[j] = arr[i]
    return arr_out


def get_file_paths(path, conf):
    """
    Get the necessary file paths for training

    :param path:      Path to the root directory where the weight files will be stored
    :param conf:      Configuration dictionary
    :param epoch_nbr: (optional) Epoch number
    :return:          (Tuple) \n
                      wFullPath:          path to the weight file \n
                      logFullPath:        path to the log file \n
                      dataAugDirPath:     path to the dataset augmentation preview directory
    """

    # path to the last saved weight file
    wFullPath = os.path.join(path, 'weights.h5')
    logFullPath = os.path.join(path, 'log.csv')
    dataAugDirPath = os.path.join(path, 'preview')

    # make data-augmentation directory if not already existing
    if not os.path.isdir(dataAugDirPath):
        os.makedirs(dataAugDirPath)

    return wFullPath, logFullPath, dataAugDirPath


def upscale_feat(xx, yy, feat_d, segm_feat):
    """
    Uses interpolation to upscale the extracted features to image resolution.
    Should be called in a multiprocessing manner.

    segm_feat: Dictionary  containing features and sp labels
    xx:        Corresponding xx-vector for interpolation
    yy:        Corresponding yy-vector for interpolation
    feat_d:    Dimension of feature
    """

    x_tmp = list()

    # iterate over images
    for im_idx in range(len(segm_feat['feat'])):
        print('   - PID: %i, process %i/%i' %
              (os.getpid(), (im_idx + 1), len(segm_feat['feat'])))

        #nbr_sp = np.max(segm_feat['segm'][im_idx]) + 1 # take zero into acount
        nbr_sp = np.unique(segm_feat['segm'][im_idx]).shape[0]

        x_new = np.arange(segm_feat['segm'][im_idx].shape[1])
        y_new = np.arange(segm_feat['segm'][im_idx].shape[0])

        xVecEntry = np.zeros((nbr_sp, feat_d)).astype('float32')

        # save all interpolated features in that array
        # having shape: (n filters, height, width)
        feat_interp = np.zeros((feat_d, segm_feat['segm'][im_idx].shape[0], segm_feat['segm'][im_idx].shape[1]))

        # iterate over feature dimensions
        for feat_idx in range(feat_d):
            # kind = ‘linear’ ‘cubic’ ‘quintic’
            f = interp2d(xx, yy, segm_feat['feat'][im_idx, :, :, feat_idx], kind='cubic')
            feat_interp[feat_idx, ...] = f(x_new, y_new)

        # iterate over superpixels
        #for sp_idx in range(nbr_sp):
        for sp_idx, i in zip(np.unique(segm_feat['segm'][im_idx]), range(nbr_sp)):
            coord_y, coord_x = np.where(segm_feat['segm'][im_idx] == sp_idx)
            xVecEntry[i, :] = np.mean(feat_interp[:, coord_y, coord_x], axis=1)

        x_tmp.append(xVecEntry)

    return x_tmp

def preprocess_imgs(im_list, post_height, post_width, nbr_channels):
    """
    Read images from list using the according mode and rescale it if necessary

    :param im_list:      Image paths as list
    :param post_height:  Requested height
    :param post_width:   Requested width
    :param nbr_channels: Number of requested color channels (1 = greyscale, 3 = RGB)
    :return:             List of images
    """

    if not isinstance(im_list, list):
        tmp = im_list
        im_list = list()
        im_list.append(tmp)

    # read image according to number of channels
    read_mode = 'RGB'
    img_dim = [len(im_list), post_height, post_width, nbr_channels]
    if nbr_channels == 1:
        read_mode = 'L'
        # if only one dimension, do not take last dimension for image processing
        img_dim = img_dim[:3]

    # prepare return array
    imgs_p = np.ndarray(img_dim, dtype=np.uint8)

    # visualize progress using progressbar
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(im_list)).start()
    for i, item in enumerate(im_list):
        img = scm.imread(item, mode=read_mode)
        if img.shape[0] == post_height and img.shape[1] == post_width:
            imgs_p[i] = img
        else:
            # interp: 'nearest' 'lanczos' 'bilinear' 'bicubic' 'cubic'
            imgs_p[i] = scm.imresize(img, (post_height, post_width, nbr_channels), interp='bilinear')
        pbar.update(i)
    pbar.finish()

    # add an additional axis if grayscale
    if nbr_channels == 1:
        imgs_p = imgs_p[..., np.newaxis]

    return imgs_p


def preprocess_and_normalize_imgs(im_list, dim_, gaze_labels=None):
    """
    Preprocess and normalize the images by providing the image path as list and optionally gaze labels

    :param im_list:     List of images as paths
    :param labels: (otional) In case of "U-Net Gaze Rec" give here the 2D-Gaussian probability maps
    :return:            (Tuple) \n
                        imgs:       Preprocessed and normalized images \n
                        imgs_mask:  Image labels squeezed to range [0,1] \n
                        mean:       Mean value over all images \n
                        std:        Standard deviation over all images \n
    """

    # preprocess images by means of loading and resizing
    inputDimY = dim_[0]
    inputDimX = dim_[1]
    nbrChannels = dim_[2]
    imgs = preprocess_imgs(im_list, inputDimY, inputDimX, nbrChannels)
    imgs = imgs.astype('float32')

    if gaze_labels == None:
        # (U-Net Reconstruct) calculate masks (masks are input images)
        imgs_mask = np.array(imgs)
        imgs_mask /= 255 # scale masks to [0,1]
    else:
        imgs_gauss = np.asarray(gaze_labels)
        imgs_gauss = imgs_gauss[..., np.newaxis]

        # (U-Net Gaze Reconstruct) add gaussian as the last axis
        imgs_mask = np.array(imgs)
        imgs_mask /= 255  # scale masks to [0,1]
        imgs_mask = np.concatenate((imgs_mask, imgs_gauss), axis=-1).astype('float32')
        del imgs_gauss

    # normalize input data
    mean = np.mean(imgs)  # mean for data centering
    std = np.std(imgs)  # std for data normalization
    imgs -= mean
    imgs /= std

    return imgs, imgs_mask, mean, std

def normalize_imgs(imgs, mean, std):

    imgs = [imgs[i, ...] for i in range(imgs.shape[0])]
    imgs = [im - mean for im in imgs]
    imgs = [im/std for im in imgs]

    return np.asarray(imgs)
