import os
import numpy as np
from skimage import io
from skimage.exposure import cumulative_distribution
import imgaug as ia
from imgaug import augmenters as iaa
from os.path import join as pjoin
import glob
from torch.utils import data
import torch
from ksptrack.models.my_augmenters import Rescale, Normalize, HistEqualize, ContrastStretching, AdaptHistEqualize
from ksptrack.models.my_augmenters import Normalize, rescale_augmenter


def base_apply_augs(sample, aug):
    truth = ia.SegmentationMapsOnImage(
        sample['label/segmentation'], shape=sample['label/segmentation'].shape)
    labels = ia.SegmentationMapsOnImage(sample['labels'],
                                        shape=sample['labels'].shape)
    im = sample['image']

    im = aug(image=im)
    truth = aug(segmentation_maps=truth).get_arr()
    labels = aug(segmentation_maps=labels).get_arr()

    sample['label/segmentation'] = truth
    sample['labels'] = labels
    sample['image'] = im

    return sample


def batch_equalize_hist_calc(image, nbins=256, mask=None):
    """Return histogram equalization parameters for a batch of images
    Parameters
    ----------
    images : list
        List of arrays
    nbins : int, optional
        Number of bins for image histogram. Note: this argument is
        ignored for integer images, for which each integer is its own
        bin.
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        are used for the equalization, which is applied to the whole image.
    Returns
    -------
    out : float array
        Image array after histogram equalization.
    Notes
    -----
    This function is adapted from [1]_ with the author's permission.
    References
    ----------
    .. [1] http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization
    """
    cdf, bin_centers = cumulative_distribution(image, nbins)
    return cdf, bin_centers


def make_normalizer(arg, img_iter):
    normalizer = iaa.Noop()

    if (isinstance(arg, iaa.Augmenter)):
        normalizer = arg
    elif (isinstance(arg, str)):
        # check for right parameters in list
        if arg not in [
                'rescale', 'std', 'rescale_histeq', 'rescale_stretching',
                'rescale_adapthist'
        ]:
            raise ValueError("Invalid value for 'normalization'")
        if (arg == 'std'):

            im_flat = np.array([im for im in img_iter])
            im_flat = im_flat.reshape((-1, 3)) / 255
            mean = im_flat.mean(axis=0)
            std = im_flat.std(axis=0)
            normalizer = iaa.Sequential(
                [rescale_augmenter,
                 Normalize(mean=mean, std=std)])
        elif (arg == 'rescale'):
            normalizer = Rescale()
        elif (arg == 'rescale_histeq'):
            normalizer = iaa.Sequential([
                iaa.contrast.CLAHE(clip_limit=3, tile_grid_size_px=3),
                rescale_augmenter
            ])
        elif (arg == 'rescale_stretching'):
            im_flat = np.array([im for im in img_iter])
            im_flat = im_flat.reshape((-1, 3))
            plow = []
            phigh = []
            for c in range(im_flat.shape[-1]):
                plow_, phigh_ = np.percentile(im_flat[..., c], (1, 99))
                plow.append(plow_)
                phigh.append(phigh_)
            normalizer = iaa.Sequential(
                [ContrastStretching(plow, phigh),
                 Rescale()])
        elif (arg == 'rescale_adapthist'):
            normalizer = iaa.Sequential([AdaptHistEqualize(0.03), Rescale()])

    return normalizer


def imread(path, scale=True):
    im = io.imread(path)

    if (im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if (scale):
        im = im / 255

    if (len(im.shape) < 3):
        im = np.repeat(im[..., None], 3, -1)

    if (im.shape[-1] > 3):
        im = im[..., 0:3]

    return im


class BaseDataset(data.Dataset):
    """
    Loads and augments images and ground truths
    """
    def __init__(self,
                 root_path,
                 augmentations=None,
                 normalization=iaa.Noop(),
                 resize_shape=None,
                 got_labels=True,
                 sp_labels_fname='sp_labels.npy'):

        self.root_path = root_path
        self.got_labels = got_labels

        exts = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for e in exts:
            img_paths.extend(
                sorted(glob.glob(pjoin(root_path, 'input-frames', e))))
        truth_paths = []
        for e in exts:
            truth_paths.extend(
                sorted(glob.glob(pjoin(root_path, 'ground_truth-frames', e))))
        self.truth_paths = truth_paths
        self.img_paths = img_paths

        if (self.got_labels):
            self.labels = np.load(pjoin(root_path, 'precomp_desc',
                                        sp_labels_fname),
                                  mmap_mode='r')
        self._normalization = make_normalizer(normalization,
                                              map(lambda s: s['image'], self))

        self._reshaper = iaa.Noop()
        self._augmentations = iaa.Noop()

        if (augmentations is not None):
            self._augmentations = augmentations

        if (resize_shape is not None):
            self._reshaper = iaa.size.Resize(resize_shape)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # When truths are resized, the values change
        # we apply a threshold to get back to binary

        im = imread(self.img_paths[idx], scale=False)

        shape = im.shape

        truth = None

        if (self.truth_paths is not None):
            truth_path = self.truth_paths[idx]
            truth = imread(truth_path, scale=False)[..., :1]
            truth = truth > 0

            if (len(truth.shape) == 3):
                truth = np.mean(truth, axis=-1)

            truth = truth.astype(np.uint8)
        else:
            truth = np.zeros(im.shape[:2]).astype(np.uint8)

        if (self.got_labels):
            labels = self.labels[idx].astype(np.int16)

        aug = iaa.Sequential(
            [self._reshaper, self._augmentations, self._normalization])
        aug_det = aug.to_deterministic()

        sample = {
            'image': im,
            'frame_name': os.path.split(self.img_paths[idx])[-1],
            'labels': labels,
            'frame_idx': idx,
            'n_frames': self.__len__(),
            'label/segmentation': truth
        }

        sample = base_apply_augs(sample, aug_det)

        return sample

    @staticmethod
    def collate_fn(data):

        to_collate = ['image', 'label/segmentation', 'labels']

        out = dict()
        for k in data[0].keys():
            if (k in to_collate):
                out[k] = torch.stack([
                    torch.from_numpy(
                        np.rollaxis(data[i][k], -1)
                        if data[i][k].ndim > 2 else data[i][k]).float()
                    for i in range(len(data))
                ])
                if out[k].ndim == 3:
                    out[k] = out[k].unsqueeze(1)
            else:
                out[k] = [data[i][k] for i in range(len(data))]

        return out


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    transf = iaa.OneOf([
        iaa.BilateralBlur(d=8, sigma_color=(100, 250), sigma_space=(100, 250)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
        iaa.GammaContrast((0.5, 2.0))
    ])

    dl = BaseDataset(pjoin('/home/ubelix/lejeune/data/medical-labeling',
                           'Dataset10'),
                     normalization='rescale',
                     augmentations=transf,
                     resize_shape=512)

    frames = np.linspace(0, len(dl) - 1, num=5, dtype=int)
    import pdb
    pdb.set_trace()  ## DEBUG ##

    ims = [dl[f]['image'] for f in frames]

    im = np.concatenate(ims, axis=1)

    plt.imshow(im)
    plt.show()
