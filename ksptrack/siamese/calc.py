from loader import Loader
from imgaug import augmenters as iaa
from my_augmenters import rescale_augmenter, Normalize
from torch.utils.data import DataLoader, SubsetRandomSampler
from siamese import Siamese
import params
import torch
import os
from os.path import join as pjoin
import utils as utls
import tqdm
from skimage.future.graph import show_rag
import numpy as np
from skimage.measure import regionprops
from skimage import future
import pandas as pd
import networkx as nx
import pickle as pk


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = Siamese(in_channels=3).to(device)
    model.eval()

    transf = iaa.Sequential([
        iaa.Resize(cfg.in_shape),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        rescale_augmenter
    ])

    dl_train = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                      augmentation=transf)

    dl = Loader(cfg.in_dir, augmentation=transf)

    dataloader = DataLoader(dl, collate_fn=dl_train.collate_fn)

    if (not os.path.exists(cfg.out_dir)):
        os.makedirs(cfg.out_dir)

    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    out_path = pjoin(cfg.out_dir, 'sp_desc_df.p')

    if (not os.path.exists(out_path)):
        print('computing features on {}'.format(cfg.in_dir))
        feats = []
        pbar = tqdm.tqdm(total=len(dataloader))

        for i, data in enumerate(dataloader):
            data = batch_to_device(data)

            feats_ = model.feat_extr(data['image']).cpu().detach().numpy()
            labels = data['labels'][0, ...].cpu().detach().numpy()

            regions = regionprops(labels)

            for props in regions:
                x, y = props.centroid
                feats.append(
                    (i, props.label, x / (labels.shape[1] - 1),
                     y / (labels.shape[0] - 1),
                     feats_[labels == props.label, :].mean(axis=0).copy()))
            pbar.update(1)
        pbar.close()
        df = pd.DataFrame(feats, columns=['frame', 'label', 'x', 'y', 'desc'])

        print('Saving features to {}'.format(out_path))
        df.to_pickle(out_path)
    else:
        print('Found features file at {}'.format(out_path))
        df = pd.read_pickle(out_path)

    if (cfg.do_inter_frame):
        out_path = pjoin(cfg.out_dir, 'graph_inter.p')

        if (not os.path.exists(out_path)):
            print('computing inter-frame probabilities on {}'.format(
                cfg.in_dir))
            pbar = tqdm.tqdm(total=len(dataloader))
            g = nx.Graph()

            for f in range(np.max(df['frame'] - 1)):

                df0 = df.loc[df['frame'] == f].copy(deep=False)
                df1 = df.loc[df['frame'] == f + 1].copy(deep=False)
                df0.columns = ['frame_0', 'label_0', 'x0', 'y0']
                df1.columns = ['frame_1', 'label_1', 'x1', 'y1']
                df0.loc[:, 'key'] = 1
                df1.loc[:, 'key'] = 1
                df_combs = pd.merge(df0, df1, on='key').drop('key', axis=1)
                df_combs['rx'] = df_combs['x0'] - df_combs['x1']
                df_combs['ry'] = df_combs['y0'] - df_combs['y1']
                r = np.concatenate((df_combs['rx'].values.reshape(
                    -1, 1), df_combs['ry'].values.reshape(-1, 1)),
                                   axis=1)
                dists = np.linalg.norm(r, axis=1)
                df_combs['dist'] = dists
                df_combs = df_combs.loc[df_combs['dist'] < cfg.radius]
                edges = [((row[1], row[2]), (row[5], row[6]))
                         for row in df_combs.itertuples()]
                feats = [
                    torch.stack((df[e[0][0], e[0][1]], df[e[0][0], e[0][1]]))
                    for e in edges
                ]
                feats = chunks(feats, cfg.batch_size)
                probas = [
                    model.calc_probas(torch.stack(feats_).to(device))
                    for feats_ in feats
                ]
                probas = [
                    p.detach().cpu().numpy().astype(np.float16) for p in probas
                ]
                edges = [((e[0][0], e[0][1]), (e[1][0], e[1][1]),
                          dict(weight=p)) for e, p in zip(edges, probas)]
                g.add_edges_from(edges)

                pbar.update(1)
            pbar.close()

            print('Saving inter-frame graph to {}'.format(out_path))
            with open(out_path, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
        else:
            print('Found inter-frame graph at {}'.format(out_path))
            with open(out_path, 'rb') as f:
                g = pk.load(f)

    if (cfg.do_intra_frame):
        out_path = pjoin(cfg.out_dir, 'graph_intra.p')

        if (not os.path.exists(out_path)):
            graphs = []
            print('computing intra-frame probabilities on {}'.format(
                cfg.in_dir))
            pbar = tqdm.tqdm(total=len(dl))

            for sample in dl:
                g = future.graph.RAG(sample['labels'])

                feats = [
                    torch.stack((df[e[0][0], e[0][1]], df[e[0][0], e[0][1]]))
                    for e in g.edges
                ]
                feats = chunks(feats, cfg.batch_size)
                probas = [
                    model.calc_probas(torch.stack(feats_).to(device))
                    for feats_ in feats
                ]
                probas = [
                    p.detach().cpu().numpy().astype(np.float16) for p in probas
                ]
                edges = [((e[0][0], e[0][1]), (e[1][0], e[1][1]),
                          dict(weight=p)) for e, p in zip(edges, probas)]
                g.add_edges_from(edges)
                graphs.append(g)
                pbar.update(1)
            pbar.close()

            print('Saving inter-frame graph to {}'.format(out_path))
            with open(out_path, 'wb') as f:
                pk.dump(graphs, f, pk.HIGHEST_PROTOCOL)
        else:
            print('Found inter-frame graph at {}'.format(out_path))
            with open(out_path, 'rb') as f:
                graphs = pk.load(f)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-dir', required=True)
    p.add('--radius', type=float, default=0.05)
    p.add('--batch-size', type=int, default=20)
    p.add('--checkpoint-path', required=True)
    p.add('--do-intra-frame', type=bool, default=True)
    p.add('--do-inter-frame', type=bool, default=True)

    cfg = p.parse_args()

    main(cfg)
