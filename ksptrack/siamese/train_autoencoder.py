from loader import Loader
from imgaug import augmenters as iaa
import logging
from my_augmenters import rescale_augmenter, Normalize
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
from skimage.future.graph import show_rag
from siamese_sp.modeling.deeplab import DeepLabv3Plus
from siamese_sp import im_utils
from skimage import color, io


class PriorMSELoss(torch.nn.Module):
    def __init__(self):
        super(PriorMSELoss, self).__init__()

    def forward(self, y, y_true, prior):

        L = ((y - y_true).pow(2) * prior).mean()

        return L


def train(cfg, model, dataloaders, run_dir, batch_to_device,
                  optimizer, logger):

    check_cp_exist = pjoin(run_dir, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if(os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    test_im_dir = pjoin(run_dir, 'recons')
    if(not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)

    criterion = PriorMSELoss()
    writer = SummaryWriter(run_dir)

    # activate RAG generation
    for dl in dataloaders.values():
        dl.dataset.do_siam_data = False
    
    best_loss = float('inf')
    for epoch in range(cfg.epochs_autoenc):

        logger.info('Epoch {}/{}'.format(epoch + 1, cfg.epochs_autoenc))

        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            if (phase in ['train', 'test']):
                # Iterate over data.
                pbar = tqdm.tqdm(total=len(dataloaders[phase]))
                for i, data in enumerate(dataloaders[phase]):
                    data = batch_to_device(data)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        im_recons, feats = model(data['image'])

                        loss = criterion(im_recons, data['image'])

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:
                            all = im_utils.make_tiled([data['image'],
                                                       im_recons])
                            io.imsave(
                                pjoin(test_im_dir, 'im_{:04d}.png'.format(epoch)),
                                all)
                            writer.add_image('{}/recons'.format(phase),
                                             all, epoch, dataformats='HWC')

                    running_loss += loss.cpu().detach().numpy()
                    loss_ = running_loss / ((i + 1) * cfg.batch_size)
                    pbar.set_description('loss: {:.4f}'.format(loss_))

                    pbar.update(1)

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
                    fname_cp='checkpoint_autoenc.pth.tar',
                    fname_bm='best_autoenc.pth.tar',
                    path=path)

def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DeepLabv3Plus()
    model.to(device)

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
                                 sampler=RandomSampler(dl_train, replacement=True,
                                                       num_samples=cfg.batch_size),
                                 collate_fn=dl_train.collate_fn,
                                 num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train, 'test': dataloader_test}

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


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
