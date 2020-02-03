import torch
import numpy as np
from tqdm import tqdm
import os
from os.path import join as pjoin
from ksptrack.siamese import im_utils
import glob
from skimage import color, io, segmentation
import utils as utls
from ksptrack.siamese.my_kmeans import MyKMeans
from ksptrack.siamese.my_agglo import MyAggloClustering, prepare_full_rag
import networkx as nx


def do_prev_clusters_init(dataloader,
                          predictions):
    # form initial cluster centres

    prev_ims = {}

    print('generating init clusters maps')
    pbar = tqdm(total=len(dataloader))
    for data in dataloader.dataset:
        labels = data['labels']
        im = data['image_unnormal']
        truth = data['label/segmentation']
        truth_cntr = segmentation.find_boundaries(np.squeeze(truth))
        im[truth_cntr, ...] = (255, 0, 0)
        all = im_utils.make_tiled_clusters(im, labels[..., 0],
                                           predictions[data['frame_idx']])
        prev_ims[data['frame_name']] = all
        pbar.update(1)
    pbar.close()

    return prev_ims


def do_prev_rags(model, device, dataloader, X):
    """
    Generate preview images on region adjacency graphs
    """

    model.eval()

    prevs = {}

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # keep only adjacent edges
        edges_to_keep = [e for e in data['graph'][0].edges()
                     if(data['graph'][0].edges[e]['adjacent'])]
        rag = data['graph'][0].edge_subgraph(edges_to_keep).copy()
        data['rag'] = [rag]

        # forward
        with torch.no_grad():
            res = model(data)

        probas = res['probas_preds'][0].detach().cpu().squeeze().numpy()
        im = data['image_unnormal'].cpu().squeeze().numpy().astype(np.uint8)
        im = np.rollaxis(im, 0, 3)
        truth = data['label/segmentation'].cpu().squeeze().numpy()
        labels = data['labels'].cpu().squeeze().numpy()

        predictions = torch.argmax(res['clusters'], dim=1).cpu().numpy()
        predictions = utls.to_onehot(predictions, res['clusters'].shape[1])
        clusters_colorized = im_utils.make_clusters(labels, predictions)
        truth = data['label/segmentation'].cpu().squeeze().numpy()
        rag_im = im_utils.my_show_rag(rag, im, labels, probas, truth=truth)
        # plot = im_utils.make_grid_rag(im,
        #                               labels,
        #                               rag,
        #                               probas,
        #                               truth=truth)
        plot = np.concatenate((im, rag_im, clusters_colorized), axis=1)
        prevs[data['frame_name'][0]] = plot

        pbar.update(1)
    pbar.close()

    return prevs


def do_prev_clusters(model, device, dataloader):

    model.eval()

    prevs = {}

    pbar = tqdm(total=len(dataloader))

    for data in dataloader:
        data = utls.batch_to_device(data, device)

        # forward
        with torch.no_grad():
            res = model(data)

        im = data['image_unnormal'].cpu().squeeze().numpy()
        im = np.rollaxis(im, 0, 3).astype(np.uint8)
        truth = data['label/segmentation'].cpu().squeeze().numpy()
        truth_cntr = segmentation.find_boundaries(truth)
        im[truth_cntr, ...] = (255, 0, 0)
        labels = data['labels'].cpu().squeeze().numpy()
        clusters = res['clusters'].cpu().squeeze().numpy()
        im = im_utils.make_tiled_clusters(im, labels, clusters)
        prevs[data['frame_name'][0]] = im

        pbar.update(1)
    pbar.close()

    return prevs

def get_features(model, dataloader, device):
    # form initial cluster centres
    features = []
    labels_pos_mask = []

    model.eval()
    model.to(device)
    print('getting features')
    pbar = tqdm(total=len(dataloader))
    for index, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)
        with torch.no_grad():
            res = model(data)

        if (len(data['labels_clicked']) > 0):
            new_labels_pos = [
                item for sublist in data['labels_clicked']
                for item in sublist
            ]
            labels_pos_mask.append([
                True if l in new_labels_pos else False
                for l in np.unique(data['labels'].cpu().numpy())
            ])
        feat = res['feats'].cpu().numpy()
        features.append(feat)
        pbar.update(1)
    pbar.close()

    return features, labels_pos_mask

def train_kmeans(model, dataloader, device, n_clusters,
                 use_locs):

    features, pos_masks = get_features(model, dataloader, device)

    print('forming {} initial clusters with kmeans (pw constraints: {})'.format(n_clusters,
                                                                                use_locs))

    kmeans = MyKMeans(n_clusters=n_clusters, use_locs=use_locs)


    # predictions = kmeans.fit_predict(features)
    print('fitting...')
    preds, init_clusters = kmeans.fit_predict(np.concatenate(features),
                                              clicked_mask=np.concatenate(pos_masks))
    predictions = [
        utls.to_onehot(p, n_clusters).ravel() for p in preds
    ]

    # split predictions by frame
    predictions = np.split(predictions,
                           np.cumsum([len(f) for f in features]))[:-1]
    return init_clusters, predictions

def train_agglo(model, dataloader, device, n_clusters, linkage):

    features, _ = get_features(model, dataloader, device)

    print('forming {} initial clusters with agglomerative clustering. Linkage: {}'.format(
        n_clusters,
        linkage))

    clf = MyAggloClustering(n_clusters, linkage)

    graphs = [s['graph'] for s in dataloader.dataset]
    labels = [s['labels'] for s in dataloader.dataset]
    full_rag = prepare_full_rag(graphs, labels)
    full_rag_mat = nx.adjacency_matrix(full_rag)

    print('fitting...')
    preds, init_clusters = clf.fit_predict(np.concatenate(features),
                                           full_rag_mat)
    predictions = [
        utls.to_onehot(p, n_clusters).ravel() for p in preds
    ]

    # split predictions by frame
    predictions = np.split(predictions,
                           np.cumsum([len(f) for f in features]))[:-1]
    return init_clusters, predictions
