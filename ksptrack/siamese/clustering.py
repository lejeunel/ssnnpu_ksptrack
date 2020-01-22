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


def do_prev_clusters_init(dataloader,
                          predictions):

    prev_ims = {}
    print('generating init clusters')
    # form initial cluster centres
    pbar = tqdm(total=len(dataloader.dataset))
    for data, preds in zip(dataloader.dataset, predictions):
        labels = data['labels']
        im = data['image_unnormal']
        all = im_utils.make_tiled_clusters(im, labels[..., 0], preds)
        prev_ims[data['frame_name']] = all
        pbar.update(1)
    pbar.close()

    return prev_ims


def do_prev_clusters(model, device, dataloader):

    model.eval()

    prevs = {}

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
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
    features = []
    labels_pos_mask = []

    model.eval()
    model.to(device)
    # form initial cluster centres
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
        pbar.update(1)
        features.append(feat)
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
