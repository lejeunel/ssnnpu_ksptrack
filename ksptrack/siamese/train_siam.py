from loader import Loader
from skimage import io
import logging
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
import torch.optim as optim
import params
import torch
import datetime
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
import tqdm
from siamese_sp.modeling.dec import DEC
from siamese_sp.modeling.siamese import Siamese
from siamese_sp.modeling.deeplab import DeepLabv3Plus
from siamese_sp import im_utils
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
import numpy as np
import matplotlib.pyplot as plt


def train(cfg, model, dataloaders, run_dir, batch_to_device,
          optimizer, logger):
    test_im_dir = pjoin(run_dir, 'dec_clusters')
    test_rag_dir = pjoin(run_dir, 'dec_rags')
    if(not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)
    if(not os.path.exists(test_rag_dir)):
        os.makedirs(test_rag_dir)

    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(run_dir)

    # activate RAG generation
    for v in dataloaders.values():
        v.dataset.do_siam_data = True

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_siam))
    for epoch in range(cfg.epochs_siam):

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  # Set model to evaluate mode
                test_ims = []
                test_rags = []

            running_loss = 0.0

            # Iterate over data.
            pbar = tqdm.tqdm(total=len(dataloaders[phase]))
            for i, data in enumerate(dataloaders[phase]):
                data = batch_to_device(data)

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    res = model(data['image'],
                                data['rag'], data['labels'])
                    loss = criterion(res['similarities'],
                                     res['similarities_labels'])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        cluster_maps = im_utils.make_label_clusters(data['labels'],
                                                                    res['clusters'])
                        all = im_utils.make_tiled([data['image'],
                                                   res['recons'],
                                                   cluster_maps])
                        test_ims.append(all)
                        test_rags.append(utls.make_grid_rag(
                            data, [r for r in res['similarities_labels']]))

                running_loss += loss.cpu().detach().numpy()
                loss_ = running_loss / ((i + 1) * cfg.batch_size)
                pbar.set_description('[{}] ep {}/{}, lss {:.4f}'.format(
                    phase,
                    epoch+1,
                    cfg.epochs_dec,
                    loss_))

                pbar.update(1)
                if phase == 'test':
                    io.imsave(pjoin(test_im_dir, 'im_{:04d}.png'.format(epoch)),
                              np.vstack(test_ims))
                    writer.add_image('{}/dec_clusters'.format(phase),
                                        np.vstack(test_ims), epoch, dataformats='HWC')
                    writer.add_figure('{}/dec_rags.'.format(phase),
                                      test_rags, epoch)
                    fig, ax = plt.subplots(len(test_rags), 2)
                    for i, (im_, lc) in enumerate(test_rags):
                        ax[i, 0].imshow(im_)
                        fig.colorbar(lc, ax=ax[i, 1], fraction=0.03)
                    fig.savefig(pjoin(test_rag_dir, 'prev_{:04d}.png'.format(epoch)),
                                dpi=400)
                        
            pbar.close()

            writer.add_scalar('{}/loss_autoenc'.format(phase),
                                loss_,
                                epoch)

            # save checkpoint
            if phase == 'test':
                is_best = False
                if (loss_ < best_loss):
                    is_best = True
                    best_loss = loss_
                path = pjoin(run_dir, 'checkpoints')
                utls.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()
                    },
                    is_best,
                    path=path)

def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    path_cp = pjoin(cfg.out_dir, 'Dataset'+cfg.train_dir,
                    'checkpoints',
                    'checkpoint_dec.pth.tar')

    if(os.path.exists(path_cp)):
        autoencoder = DeepLabv3Plus()
        dec = DEC(cluster_number=cfg.n_clusters,
                  hidden_dimension=304,
                  autoencoder=autoencoder)
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        dec.load_state_dict(state_dict)
        dec.to(device)
    else:
        print('checkpoint {} not found. Train DEC first'.format(path_cp))
        return

    model = Siamese(dec,
                    n_edges=cfg.n_edges).to(device)

    transf, transf_normal = im_utils.make_data_aug(cfg)

    dl_train = Loader(pjoin(cfg.in_root, 'Dataset'+cfg.train_dir),
                      augmentation=transf,
                      n_segments=cfg.n_segments_train,
                      delta_segments=cfg.delta_segments_train,
                      normalization=transf_normal)

    dataloader_train = DataLoader(dl_train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl_train.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)
    dataloader_test = DataLoader(dl_train,
                                 batch_size=cfg.batch_size,
                                 sampler=RandomSampler(dl_train,
                                                       replacement=True,
                                                       num_samples=cfg.n_batches_test),
                                 collate_fn=dl_train.collate_fn,
                                 drop_last=True,
                                 num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train,
                   'test': dataloader_test}

    ds_dir = os.path.split('Dataset'+cfg.train_dir)[-1]

    run_dir = pjoin(cfg.out_dir, '{}'.format(ds_dir))

    if(not os.path.exists(run_dir)):
        os.makedirs(run_dir)

    # Save cfg
    with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    optimizer = optim.SGD(params=[
        {'params': model.parameters(), 'lr': cfg.lr_autoenc}
    ],
                          momentum=cfg.momentum,
                          weight_decay=cfg.decay)

    utls.setup_logging(run_dir)
    logger = logging.getLogger('siam')

    logger.info('run_dir: {}'.format(run_dir))

    train(cfg, model, dataloaders,
          run_dir, batch_to_device, optimizer, logger)

    logger.info('training siam')


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
