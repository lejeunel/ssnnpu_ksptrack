import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io, color, segmentation, draw
from imgaug import augmenters as iaa
from ksptrack.models.my_augmenters import rescale_augmenter
from ksptrack.pu import utils as utls
from skimage.future.graph import show_rag
import tqdm
import torch.nn as nn
from ksptrack.pu.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
import pandas as pd


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
    features = []
    features_bagger = []
    outputs = []
    outputs_unpooled = []
    truths = []
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
            data['pos_labels'][data['pos_labels']['frame'] == f_] for f_ in f
        ])
        pos_labels = pos_labels.dropna()
        unq_labels = np.unique(data['labels'][0].cpu().numpy())
        to_add = np.zeros(unq_labels.shape[0]).astype(bool)
        if not data['pos_labels'].empty:

            clicked_labels = tuple(data['pos_labels']['label'])
            idx_clicked = [
                np.argwhere(unq_labels == c) for c in clicked_labels
            ]
            to_add[np.array(idx_clicked).flatten()] = True
        labels_pos_mask.append(to_add)

        f = sp_pool(res['feats'], data['labels'])
        out = sp_pool(res['output'].sigmoid(), data['labels'])
        out_unpooled = res['output'].sigmoid()
        truth = sp_pool(data['label/segmentation'], data['labels'])
        truth = truth >= 0.5

        upsamp = nn.UpsamplingBilinear2d(data['labels'].size()[-2:])
        coords = make_coord_map(1, data['labels'].shape[3],
                                data['labels'].shape[2]).to(device)
        prior = upsamp(data['loc_prior'])
        f_bag = upsamp(res['feats'])
        # f_bag = torch.cat((f_bag, coords, prior), dim=1)
        f_bag = torch.cat((f_bag, coords), dim=1)
        f_bag = sp_pool(f_bag, data['labels'])

        truths.append(truth.detach().cpu().numpy().squeeze())
        features.append(f.detach().cpu().numpy().squeeze())
        features_bagger.append(f_bag.detach().cpu().numpy().squeeze())
        outputs.append(out.detach().cpu().numpy().squeeze())
        outputs_unpooled.append(out_unpooled.detach().cpu().numpy().squeeze())
        pbar.set_description('[fwd pass]')
        pbar.update(1)
    pbar.close()

    res = [features, labels_pos_mask]
    res = {
        'feats': features,
        'feats_bag': features_bagger,
        'outs': outputs,
        'outs_unpooled': outputs_unpooled,
        'labels_pos_mask': labels_pos_mask,
        'pos': positions,
        'truths': truths
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


def make_grid_samples(batch, edges_pw, n_clusters):
    res = []
    for i in range(len(batch['graph'])):
        im = batch['image'][i].cpu().detach().numpy()
        im = (255 * im).astype(np.uint8)
        im = np.rollaxis(im, 0, 3)
        labels = batch['labels'][i].squeeze().cpu().detach().numpy()
        graph = batch['graph'][i]
        predictions = np.array(
            [graph.nodes[n]['cluster'] for n in graph.nodes()])
        predictions = utls.to_onehot(predictions, n_clusters)

        tile = make_tiled_clusters(im, labels, predictions)
        im_graph = show_sampled_edges(im, labels.astype(int), graph,
                                      edges_pw[i])
        res.append(np.concatenate((tile, im_graph), axis=1))

    return np.concatenate(res, axis=0)


def my_show_rag(graph, image, labels, probas, truth=None):

    cmap = plt.get_cmap('viridis')
    centroids = np.array([graph.nodes[n1]['centroid'][::-1] + \
                          graph.nodes[n2]['centroid'][::-1]
                          for n1, n2 in graph.edges()])

    cvt = np.array(
        [labels.shape[0], labels.shape[1], labels.shape[0], labels.shape[1]])
    centroids *= cvt
    centroids = centroids.astype(int)

    lines = [np.array(draw.line(*r)) for r in centroids]

    colors = np.concatenate([
        np.repeat((np.array(cmap(p)[:3]))[..., None], l.shape[1], axis=1)
        for l, p in zip(lines, probas)
    ],
                            axis=1)
    colors = (colors * 255).astype(int)

    lines = np.concatenate(lines, axis=1)

    out = image.copy()
    out[segmentation.find_boundaries(labels), ...] = (0, 0, 0)
    if (truth is not None):
        out[segmentation.find_boundaries(truth), ...] = (255, 0, 0)

    out[lines[0], lines[1], :] = colors.T

    return out


def make_grid_rag(im, labels, rag, probas, truth=None):

    fig = plt.figure()
    ax = plt.gca()

    # make preview images
    rag.add_edges_from([(n0, n1, dict(weight=probas[j]))
                        for j, (n0, n1) in enumerate(rag.edges())])

    lc = show_rag(labels.astype(int),
                  rag,
                  im,
                  ax=ax,
                  edge_width=0.5,
                  edge_cmap='viridis')
    fig.colorbar(lc, ax=ax, fraction=0.03)
    if (truth is not None):
        truth_contour = segmentation.find_boundaries(truth, mode='thick')
        im[truth_contour, ...] = (255, 0, 0)
    ax.axis('off')
    ax.imshow(im)

    fig.tight_layout(pad=0, w_pad=0)
    fig.canvas.draw()
    im_plot = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    plt.close(fig)
    return im_plot


def make_clusters(labels, predictions):
    cmap = plt.get_cmap('viridis')
    shape = labels.shape
    n_clusters = predictions.shape[1]
    mapping = np.array([
        (np.array(cmap(c / n_clusters)[:3]) * 255).astype(np.uint8)
        for c in np.argmax(predictions, axis=1)
    ])
    mapping = np.concatenate((np.unique(labels)[..., None], mapping), axis=1)

    _, ind = np.unique(labels, return_inverse=True)
    clusters_colorized = mapping[ind, 1:].reshape((shape[0], shape[1], 3))
    clusters_colorized = clusters_colorized.astype(np.uint8)

    return clusters_colorized


def make_tiled_clusters(im, labels, predictions):

    clusters_colorized = make_clusters(labels, predictions)

    return np.concatenate((im, clusters_colorized), axis=1)


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


def make_label_clusters(labels_batch, clusters_batch):

    ims = []

    if (isinstance(labels_batch, torch.Tensor)):
        labels_batch = labels_batch.squeeze(1).cpu().numpy()

    if (isinstance(clusters_batch, torch.Tensor)):
        n_labels = [np.unique(labs).size for labs in labels_batch]
        clusters_batch = torch.split(clusters_batch, n_labels, dim=0)

    for labels, clusters in zip(labels_batch, clusters_batch):
        map_ = np.zeros(labels.shape[:2]).astype(np.uint8)
        for i, l in enumerate(np.unique(labels)):
            map_[labels == l] = torch.argmax(clusters[i, :]).cpu().item()
        ims.append(color.label2rgb(map_))

    return ims


def overlap_contours(imgs, labels):
    ims = []
    for im, lab in zip(imgs, labels):
        im = im.detach().cpu().numpy().transpose((1, 2, 0))
        lab = lab.detach().cpu().numpy().transpose((1, 2, 0))[..., 0]
        lab_contour = segmentation.find_boundaries(lab, mode='thick')
        im[lab_contour, :] = (1, 0, 0)
        ims.append(im)

    return ims


def make_tiled(tnsr_list, rescale=True):
    """
    tnsr_list: list of tensors
    modes iterable of strings ['image', 'map']
    """

    if (not isinstance(rescale, list)):
        rescale = [True] * len(tnsr_list)

    arr_list = list()
    for i, t in enumerate(tnsr_list):
        if (isinstance(t, torch.Tensor)):
            t = np.vstack([img_tensor_to_img(t_, rescale[i]) for t_ in t])
        else:
            t = np.vstack([t_ for t_ in t])

        arr_list.append(t)

    all = np.concatenate(arr_list, axis=1)
    all = (all * 255).astype(np.uint8)

    return all


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
