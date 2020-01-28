import torch
import numpy as np
from tqdm import tqdm
import os
from os.path import join as pjoin
from ksptrack.siamese import im_utils
import glob
from skimage import color, io
import utils as utls
from ksptrack.siamese.my_kmeans import MyKMeans
from ksptrack.siamese.my_agglo import MyAggloClustering, prepare_full_rag
import networkx as nx


def do_prev_clusters_init(dataloader,
                          predictions,
                          frames=[]):
    # form initial cluster centres

    prev_ims = {}
    if(len(frames) == 0):
        frames = [i for i in range(len(dataloader))]

    print('generating init clusters')
    pbar = tqdm(total=len(dataloader))
    for i, (data, preds) in enumerate(zip(dataloader.dataset, predictions)):
        if(i in frames):
            labels = data['labels']
            im = data['image_unnormal']
            all = im_utils.make_tiled_clusters(im, labels[..., 0], preds)
            prev_ims[data['frame_name']] = all
        pbar.update(1)
    pbar.close()

    return prev_ims


def do_prev_rags(model, device, dataloader, frames=[]):

    model.eval()

    prevs = {}

    if(len(frames) == 0):
        frames = [i for i in range(len(dataloader))]

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        if(i in frames):
            data = utls.batch_to_device(data, device)

            # forward
            with torch.no_grad():
                res = model(data)

            graph = data['graph'][0]
            to_remove = [e for e in graph.edges() if(not graph.edges[e]['adjacent'])]
            graph.remove_edges_from(to_remove)
            probas = model.calc_all_probas(res['feats'], graph)
            probas = probas.detach().cpu().numpy()
            im = data['image_unnormal'].cpu().squeeze().numpy().astype(np.uint8)
            im = np.rollaxis(im, 0, 3)
            labels = data['labels'].cpu().squeeze().numpy()
            truth = data['label/segmentation'].cpu().squeeze().numpy()
            plot = im_utils.make_grid_rag(im,
                                        labels,
                                        graph,
                                        probas,
                                        truth=truth)
            prevs[data['frame_name'][0]] = plot

        pbar.update(1)
    pbar.close()

    return prevs


def do_prev_clusters(model, device, dataloader, frames=[]):

    model.eval()

    prevs = {}

    if(len(frames) == 0):
        frames = [i for i in range(len(dataloader))]

    pbar = tqdm(total=len(dataloader))

    for i, data in enumerate(dataloader):
        if(i in frames):
            data = utls.batch_to_device(data, device)

            # forward
            with torch.no_grad():
                res = model(data)

            im = data['image_unnormal'].cpu().squeeze().numpy()
            im = np.rollaxis(im, 0, 3).astype(np.uint8)
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
