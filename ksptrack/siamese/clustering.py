import torch
import numpy as np
from tqdm import tqdm
from ksptrack.siamese import im_utils
from ksptrack.siamese import utils as utls
from skimage import segmentation


def do_prev_clusters_init(dataloader, predictions, probas=None):
    # form initial cluster centres

    prev_ims = {}

    print('generating init clusters maps')
    pbar = tqdm(total=len(dataloader))
    for data in dataloader.dataset:
        labels = data['labels']
        im = (data['image'] * 255).astype(np.uint8)
        truth = data['label/segmentation']
        truth_cntr = segmentation.find_boundaries(np.squeeze(truth))
        im[truth_cntr, ...] = (255, 0, 0)
        all = im_utils.make_tiled_clusters(im, labels[..., 0],
                                           predictions[data['frame_idx']])
        prev_ims[data['frame_name']] = all
        pbar.update(1)
    pbar.close()

    return prev_ims


def do_prev_rags(model, device, dataloader, couple_graphs):
    """
    Generate preview images on region adjacency graphs
    """

    model.eval()

    prevs = {}

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # keep only adjacent edges
        edges_rag = [
            e for e in data['graph'][0].edges()
            if (data['graph'][0].edges[e]['adjacent'])
        ]
        rag = data['graph'][0].edge_subgraph(edges_rag).copy()

        # forward
        with torch.no_grad():
            res = model(data, torch.tensor(edges_rag))

        probas = res['probas_preds'].detach().cpu().squeeze().numpy()
        im = data['image'].cpu().squeeze().numpy().astype(np.uint8)
        im = (255 * im).astype(np.uint8)
        im = np.rollaxis(im, 0, 3)
        truth = data['label/segmentation'].cpu().squeeze().numpy()
        labels = data['labels'].cpu().squeeze().numpy()

        predictions = couple_graphs.nodes[data['frame_idx']
                                          [0]]['clst'].cpu().numpy()
        predictions = utls.to_onehot(predictions, res['clusters'].shape[1])
        clusters_colorized = im_utils.make_clusters(labels, predictions)
        truth = data['label/segmentation'].cpu().squeeze().numpy()
        rag_im = im_utils.my_show_rag(rag, im, labels, probas, truth=truth)
        plot = np.concatenate((im, rag_im, clusters_colorized), axis=1)
        prevs[data['frame_name'][0]] = plot

        pbar.update(1)
    pbar.close()

    return prevs


def do_prev_clusters(model, device, dataloader, *args):

    model.eval()
    model.to(device)

    prevs = {}

    pbar = tqdm(total=len(dataloader))

    for data in dataloader:
        data = utls.batch_to_device(data, device)

        # forward
        with torch.no_grad():
            res = model(data, *args)

        im = data['image'].cpu().squeeze().numpy()
        im = (255 * im).astype(np.uint8)
        im = np.rollaxis(im, 0, 3)
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


def get_features(model,
                 dataloader,
                 device,
                 return_distribs=False,
                 return_obj_preds=False,
                 feat_fields=['pooled_feats', 'proj_pooled_feats']):

    if (isinstance(feat_fields, str)):
        feat_fields = [feat_fields]

    # form initial cluster centres
    labels_pos_mask = dict()
    distribs = dict()
    features = {k: dict() for k in feat_fields}
    obj_preds = dict()

    sigmoid = torch.nn.Sigmoid()
    model.eval()
    model.to(device)

    pbar = tqdm(total=len(dataloader))
    for index, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)
        with torch.no_grad():
            res = model(data)

        start = 0
        for i, f in enumerate(data['frame_idx']):
            end = start + torch.unique(data['labels'][i]).numel()
            obj_preds[f] = sigmoid(
                res['rho_hat_pooled'][start:end]).detach().cpu().numpy()
            for k in feat_fields:
                features[k][f] = res[k][start:end].detach().cpu().numpy()
            distribs[f] = res['clusters'][start:end].detach().cpu().numpy()

            clicked_labels = tuple(data['pos_labels'][
                data['pos_labels']['frame'] == f]['label'].tolist())
            unq_labels = np.unique(data['labels'][i].cpu().numpy())
            idx_clicked = [
                np.argwhere(unq_labels == c) for c in clicked_labels
            ]
            to_add = np.zeros(unq_labels.shape[0]).astype(bool)
            to_add[tuple(idx_clicked)] = True
            labels_pos_mask[f] = to_add

            start += end

        pbar.update(1)
    pbar.close()

    obj_preds = [obj_preds[k] for k in sorted(obj_preds.keys())]
    features = {
        k: [features[k][i] for i in sorted(features[k].keys())]
        for k in feat_fields
    }
    distribs = [distribs[k] for k in sorted(distribs.keys())]
    labels_pos_mask = [
        labels_pos_mask[k] for k in sorted(labels_pos_mask.keys())
    ]

    res = [features, labels_pos_mask]

    if (return_distribs):
        res.append(np.concatenate(distribs))

    if (return_obj_preds):
        res.append(obj_preds)

    return res
