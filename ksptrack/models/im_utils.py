import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
import os
from skimage import io
from imgaug.augmenters import Augmenter
from multiprocessing import Pool


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def coord2Pixel(x, y, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = int(np.round(x * (width - 1), 0))
    i = int(np.round(y * (height - 1), 0))

    return i, j


def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5 * ((x - x0) / std)**2)

    return y / np.sum(y)


def make_2d_gauss(shape, std, center):
    """
    Make object prior (gaussians) on center
    """

    g = np.zeros(shape)
    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1, 1), (1, shape[1]))

    g = g_x * g_y

    return g / np.sum(g)


def upsample_sequence(arr, out_shape, fnames, order=1, n_jobs=1):
    """
    Wrapper for upsample with or without parallel processes
    """

    args_ = [(i, a, out_shape, order, fnames[i]) for i, a in enumerate(arr)]
    if (n_jobs > 1):
        args_ = utls.chunks(args_, len(args_) // n_jobs)

        pool = Pool()
        out = pool.map(upsample_batch, args_)

    else:
        out = upsample_batch(args_)

    return out


def normalize_img(img, mean, std):

    return (img - mean) / std


class PixelScaling(Augmenter):
    """
    Scales values between 0 and 1
    """

    def __init__(self,
                 name=None,
                 deterministic=True,
                 random_state=None):
        super(PixelScaling, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.name = "pixelscaling"

    def _augment_images(self, images, random_state, parents, hooks):
        return [(im - im.min())/(im.max() - im.min()) for im in images]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []

class NormalizeAug(Augmenter):
    """

    """

    def __init__(self,
                 mean,
                 std,
                 name=None,
                 deterministic=True,
                 random_state=None):
        super(NormalizeAug, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.name = "normalize"
        self.mean = mean
        self.std = std

    def _augment_images(self, images, random_state, parents, hooks):
        return [normalize_img(im, self.mean, self.std) for im in images]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


def imread(path, scale=True):
    im = io.imread(path)
    if (scale):
        im = im / 255

    if(im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if(len(im.shape) < 3):
        im = np.repeat(im[..., None], 3, -1)

    if (im.shape[-1] > 3):
        im = im[..., 0:3]

    return im


def img_tensor_to_img(tnsr, rescale=False):
    im = tnsr.detach().cpu().numpy().transpose((1, 2, 0))
    if (rescale):
        range_ = im.max() - im.min()
        im = (im - im.min()) / range_
    return im


def one_channel_tensor_to_img(tnsr, rescale=False):
    im = tnsr.detach().cpu().numpy().transpose((1, 2, 0))
    if (rescale):
        range_ = im.max() - im.min()
        im = (im - im.min()) / range_
    im = np.repeat(im, 3, axis=-1)
    return im


def save_tensors(tnsr_list, kind, path, rescale=True):
    """
    tnsr_list: list of tensors
    modes iterable of strings ['image', 'map']
    """

    if (not isinstance(rescale, list)):
        rescale = [True] * len(tnsr_list)

    path_ = os.path.split(path)[0]
    if (not os.path.exists(path_)):
        os.makedirs(path_)

    arr_list = list()
    for i, t in enumerate(tnsr_list):
        if (kind[i] == 'image'):
            arr_list.append(img_tensor_to_img(t, rescale[i]))
        elif (kind[i] == 'map'):
            arr_list.append(one_channel_tensor_to_img(t, rescale[i]))
        else:
            raise Exception("kind must be 'image' or 'map'")

    all = np.concatenate(arr_list, axis=1)
    io.imsave(path, all)


def center_crop(x, center_crop_size):
    assert x.ndim == 3
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:, centerw - halfw:centerw + halfw, centerh - halfh:centerh +
             halfh]


def to_np_array(x):
    return x.numpy().transpose((1, 2, 0))


def to_tensor(x):
    import torch

    # copy to avoid "negative stride numpy" problem
    x = x.transpose((2, 0, 1)).copy()
    return torch.from_numpy(x).float()


def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret


def poisson_downsampling(image, peak, random_state=np.random):
    if not isinstance(image, np.ndarray):
        imgArr = np.array(image, dtype='float32')
    else:
        imgArr = image.astype('float32')
    Q = imgArr.max(axis=(0, 1)) / peak
    if Q[0] == 0:
        return imgArr
    ima_lambda = imgArr / Q
    noisy_img = random_state.poisson(lam=ima_lambda)
    return noisy_img.astype('float32')


def elastic_transform(image,
                      alpha=1000,
                      sigma=30,
                      spline_order=1,
                      mode='nearest',
                      random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
        cval=0) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
        cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order,
            mode=mode).reshape(shape)
    return result
