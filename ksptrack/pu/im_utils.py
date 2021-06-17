import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from imgaug import augmenters as iaa
from ksptrack.utils.my_augmenters import rescale_augmenter
from ksptrack.pu import utils as utls
from skimage import color, draw, io, segmentation
from skimage.future.graph import show_rag
from superpixPool.superpixPool_layer import SupPixPool


def sp_pool(feats, labels):
    roi_pool = SupPixPool()
    upsamp = nn.UpsamplingBilinear2d(labels.size()[-2:])
    pooled_feats = []
    for b in range(labels.shape[0]):
        labels_ = labels[b]
        pooled_feats_ = roi_pool(upsamp(feats[b].unsqueeze(0)),
                                 labels_).squeeze().T
        pooled_feats.append(pooled_feats_)
    return torch.cat(pooled_feats)


def make_coord_map(batch_size, w, h, return_tuple=False):
    xx_ones = torch.ones([1, 1, 1, w], dtype=torch.int32)
    yy_ones = torch.ones([1, 1, 1, h], dtype=torch.int32)

    xx_range = torch.arange(w, dtype=torch.int32)
    yy_range = torch.arange(h, dtype=torch.int32)
    xx_range = xx_range[None, None, :, None]
    yy_range = yy_range[None, None, :, None]

    xx_channel = torch.matmul(xx_range, xx_ones)
    yy_channel = torch.matmul(yy_range, yy_ones)

    # transpose y
    yy_channel = yy_channel.permute(0, 1, 3, 2)

    xx_channel = xx_channel.float() / (w - 1)
    yy_channel = yy_channel.float() / (h - 1)

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

    if return_tuple:
        return xx_channel, yy_channel
    else:
        return torch.cat([xx_channel, yy_channel], dim=1)


def get_positions_labels(labels):
    coord_xx, coord_yy = make_coord_map(labels.shape[0],
                                        labels.shape[2],
                                        labels.shape[3],
                                        return_tuple=True)

    coord_xx = coord_xx.to(labels.device)
    coord_yy = coord_yy.to(labels.device)

    pos_x = sp_pool(coord_xx, labels)
    pos_y = sp_pool(coord_yy, labels)

    res = {'pos_x': pos_x, 'pos_y': pos_y}

    return res


def get_features(model, dataloader, device, loc_prior=False):
    # form initial cluster centres
    labels_pos_mask = []
    outputs = []
    outputs_unpooled = []
    truths = []
    truths_unpooled = []
    positions = []

    model.eval()
    model.to(device)
    # print('getting features')
    pbar = tqdm.tqdm(total=len(dataloader))
    for index, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)
        with torch.no_grad():
            if loc_prior:
                res = model(
                    torch.cat((data['image'], data['loc_prior']), dim=1))
            else:
                res = model(data['image'])

        f = data['frame_idx']

        pos_ = get_positions_labels(data['labels'])
        pos_ = torch.cat((pos_['pos_x'][..., None], pos_['pos_y'][..., None]),
                         dim=1)
        pos_ = pos_.detach().cpu().numpy()
        positions.append(pos_)

        pos_labels = pd.concat([
            data['annotations'][data['annotations']['frame'] == f_]['label']
            for f_ in f
        ])
        pos_labels = pos_labels.dropna()
        unq_labels = np.unique(data['labels'][0].cpu().numpy())
        to_add = np.zeros(unq_labels.shape[0]).astype(bool)
        if not data['annotations'].empty:

            clicked_labels = tuple(data['annotations']['label'])
            idx_clicked = [
                np.argwhere(unq_labels == c) for c in clicked_labels
            ]
            to_add[np.array(idx_clicked).flatten()] = True
        labels_pos_mask.append(to_add)

        out = sp_pool(res['output'].sigmoid(), data['labels'])
        out_unpooled = res['output'].sigmoid()
        truth = sp_pool(data['label/segmentation'], data['labels'])
        truth = truth >= 0.5
        truth_unpooled = data['label/segmentation'] >= 0.5

        upsamp = nn.UpsamplingBilinear2d(data['labels'].size()[-2:])
        coords = make_coord_map(1, data['labels'].shape[3],
                                data['labels'].shape[2]).to(device)

        truths_unpooled.append(truth_unpooled.detach().cpu().numpy().squeeze())
        truths.append(truth.detach().cpu().numpy().squeeze())
        outputs.append(out.detach().cpu().numpy().squeeze())
        outputs_unpooled.append(out_unpooled.detach().cpu().numpy().squeeze())
        pbar.set_description('[fwd pass]')
        pbar.update(1)
    pbar.close()

    res = {
        'outs': outputs,
        'outs_unpooled': outputs_unpooled,
        'labels_pos_mask': labels_pos_mask,
        'pos': positions,
        'truths': truths,
        'truths_unpooled': truths_unpooled
    }

    return res


def colorize(map_):
    cmap = plt.get_cmap('viridis')
    map_colorized = (cmap(map_)[..., :3] * 255).astype(np.uint8)

    return map_colorized


def show_sampled_edges(image, labels, graph, edges_pw):
    """Show a Region Adjacency Graph on an image.
    """

    out = image.copy()
    out[segmentation.find_boundaries(labels), :] = (0, 0, 255)
    # Defining the end points of the edges
    # The tuple[::-1] syntax reverses a tuple as matplotlib uses (x,y)
    # convention while skimage uses (row, column)
    pos_edges = edges_pw.loc[edges_pw['clust_sim'] == 1].to_numpy()[:, :2]
    neg_edges = edges_pw.loc[edges_pw['clust_sim'] == 0].to_numpy()[:, :2]

    pos_lines = np.array([graph.nodes[n1]['centroid'][::-1] + \
                          graph.nodes[n2]['centroid'][::-1]
                          for n1, n2 in pos_edges])
    neg_lines = np.array([graph.nodes[n1]['centroid'][::-1] + \
                          graph.nodes[n2]['centroid'][::-1]
                          for n1, n2 in neg_edges])

    cvt = np.array(
        [labels.shape[0], labels.shape[1], labels.shape[0], labels.shape[1]])
    pos_lines *= cvt
    neg_lines *= cvt
    pos_lines = pos_lines.astype(int)
    neg_lines = neg_lines.astype(int)

    pos_lines = np.concatenate([draw.line(*r) for r in pos_lines], axis=1)
    neg_lines = np.concatenate([draw.line(*r) for r in neg_lines], axis=1)

    out[pos_lines[0], pos_lines[1], :] = (0, 255, 0)
    out[neg_lines[0], neg_lines[1], :] = (255, 0, 0)

    return out


def make_data_aug(cfg, do_resize=False):
    transf = iaa.Sequential([
        iaa.BilateralBlur(d=8,
                          sigma_color=(cfg.aug_blur_color_low,
                                       cfg.aug_blur_color_high),
                          sigma_space=(cfg.aug_blur_space_low,
                                       cfg.aug_blur_space_high)),
        iaa.Affine(scale={
            "x": (1 - cfg.aug_scale, 1 + cfg.aug_scale),
            "y": (1 - cfg.aug_scale, 1 + cfg.aug_scale)
        },
                   rotate=(-cfg.aug_rotate, cfg.aug_rotate),
                   shear=(-cfg.aug_shear, cfg.aug_shear)),
        iaa.AdditiveGaussianNoise(scale=(0, cfg.aug_noise * 255)),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
    ])

    transf_normal = iaa.Sequential([rescale_augmenter])

    if (do_resize):
        transf_normal.add(iaa.size.Resize(cfg.in_shape))

    return transf, transf_normal


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


def imread(path, scale=True):
    im = io.imread(path)
    if (scale):
        im = im / 255

    if (im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if (len(im.shape) < 3):
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


def center_crop(x, center_crop_size):
    assert x.ndim == 3
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:, centerw - halfw:centerw + halfw,
             centerh - halfh:centerh + halfh]


def to_np_array(x):
    return x.numpy().transpose((1, 2, 0))


def to_tensor(x):
    import torch

    # copy to avoid "negative stride numpy" problem
    x = x.transpose((2, 0, 1)).copy()
    return torch.from_numpy(x).float()
