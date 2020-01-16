from loader import Loader
from imgaug import augmenters as iaa
import logging
from my_augmenters import rescale_augmenter, Normalize
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from siamese import Siamese
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

def sample_edges(rag, feats):
    edges = [e for e in rag.edges]
    dists = []

def adjust_lr(optimizer, itr, max_itr, lr_autoenc, lr_siam, pwr):
    now_lr_autoenc = lr_autoenc * (1 - itr/(max_itr+1)) ** pwr
    now_lr_siam = lr_autoenc * (1 - itr/(max_itr+1)) ** pwr
    optimizer.param_groups[0]['lr'] = now_lr_autoenc
    optimizer.param_groups[1]['lr'] = now_lr_autoenc
    optimizer.param_groups[2]['lr'] = now_lr_siam
    optimizer.param_groups[3]['lr'] = now_lr_siam

    return now_lr_autoenc, now_lr_siam

def train(cfg, model, dataloaders, run_dir, batch_to_device,
                  optimizer, logger):
    test_rag_dir = pjoin(run_dir, 'rags')
    if(not os.path.exists(test_rag_dir)):
        os.makedirs(test_rag_dir)
    test_im_dir = pjoin(run_dir, 'recons')
    if(not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)

    criterion_siam = torch.nn.BCEWithLogitsLoss()
    criterion_autoenc = torch.nn.MSELoss()
    writer = SummaryWriter(run_dir)

    # activate RAG generation
    for dl in dataloaders.values():
        dl.dataset.do_siam_data = True
    
    best_loss = float('inf')
    for epoch in range(cfg.epochs):

        logger.info('Epoch {}/{}'.format(epoch + 1, cfg.epochs))

        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss_siam = 0.0
            running_loss_autoenc = 0.0

            if (phase in ['train', 'test']):
                # Iterate over data.
                loss_autoenc_log = tqdm.tqdm(total=0,
                                             position=0,
                                             bar_format='{desc}')
                loss_siam_log = tqdm.tqdm(total=0,
                                          position=1,
                                          bar_format='{desc}')
                loss_total_log = tqdm.tqdm(total=0,
                                           position=2,
                                           bar_format='{desc}')
                pbar = tqdm.tqdm(total=len(dataloaders[phase]), position=3)
                for i, data in enumerate(dataloaders[phase]):
                    data = batch_to_device(data)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    edges_to_pool = None

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if(phase == 'test'):
                            edges_to_pool = [[e for e in g.edges] for g in data['rag']]
                            res = model(data['image'], data['rag'], data['labels'],
                                           edges_to_pool)
                            res_ = torch.cat(res['similarities'])
                            y_ = torch.cat(res['similarities_labels'])
                        else:
                            im_recons, feats = model.autoenc(data['image'])
                            feats_sps = model.sp_pool(
                            res = model(data['image'], data['nn_graph'], data['labels'])
                            res_ = torch.stack(res['similarities'])
                            y_ = torch.stack(res['similarities_labels'])

                        loss_siam = criterion_siam(res_, y_)
                        loss_autoenc = criterion_autoenc(res['recons'], data['image'])

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss = cfg.gamma * loss_siam + (1 - cfg.gamma) * loss_autoenc
                            loss.backward()
                            optimizer.step()
                        else:
                            fig = utls.make_grid_rag(data,
                                                     [r for r in res['similarities']])
                            writer.add_figure('{}/edges'.format(phase),
                                              fig, epoch)
                            fig.savefig(pjoin(test_rag_dir, 'prev_{:04d}.png'.format(epoch)),
                                        dpi=400)
                            all = im_utils.save_tensors([data['image'][0],
                                                         res['recons'][0]],
                                                        ['image', 'image'],
                                                        pjoin(test_im_dir,
                                                              'im_{:04d}.png'.format(epoch)))
                            writer.add_image('{}/recons'.format(phase),
                                             all, epoch, dataformats='HWC')

                    running_loss_siam += loss_siam.cpu().detach().numpy()
                    loss_siam_ = running_loss_siam / ((i + 1) * cfg.batch_size)
                    running_loss_autoenc += loss_autoenc.cpu().detach().numpy()
                    loss_autoenc_ = running_loss_autoenc / ((i + 1) * cfg.batch_size)

                    loss_autoenc_log.set_description_str('loss recons: {:.4f}'.format(loss_autoenc_))
                    loss_siam_log.set_description_str('loss edges: {:.4f}'.format(loss_siam_))
                    loss_total_log.set_description_str('loss total: {:.4f}'.format(loss_autoenc_ + loss_siam_))
                    pbar.update(1)

                pbar.close()
                loss_autoenc_log.close()
                loss_siam_log.close()
                loss_total_log.close()
                writer.add_scalar('{}/loss_edges'.format(phase),
                                  loss_siam_,
                                  epoch)
                writer.add_scalar('{}/loss_recons'.format(phase),
                                  loss_autoenc_,
                                  epoch)
                writer.add_scalar('{}/loss_total'.format(phase),
                                  loss_autoenc_ + loss_siam_,
                                  epoch)

            # save checkpoint
            if phase == 'test':
                is_best = False
                if (loss_autoenc_ + loss_siam_ < best_loss):
                    is_best = True
                    best_loss = loss_autoenc + loss_siam_
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

    autoenc = DeepLabv3Plus()
    model = Siamese(autoenc,
                    in_channels=3,
                    n_edges=cfg.n_edges,
                    sp_pool_use_max=cfg.sp_pooling_max)
    if(cfg.checkpoint_autoenc is not None):
        print('loading checkpoint {}'.format(cfg.checkpoint_autoenc))
        state_dict = torch.load(cfg.checkpoint_autoenc,
                                map_location=lambda storage, loc: storage)
        autoenc.load_state_dict(state_dict)
    elif(cfg.checkpoint_siam is not None):
        print('loading checkpoint {}'.format(cfg.checkpoint_siam))
        state_dict = torch.load(cfg.checkpoint_siam,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    autoenc.to(device)
    model.to(device)

    transf = iaa.Sequential([
        iaa.Invert(0.5) if 'Dataset1' in 'Dataset'+cfg.train_dir else iaa.Noop(),
        iaa.SomeOf(3,
                    [iaa.Affine(
                        scale={
                            "x": (1 - cfg.aug_scale,
                                    1 + cfg.aug_scale),
                            "y": (1 - cfg.aug_scale,
                                    1 + cfg.aug_scale)
                        },
                        rotate=(-cfg.aug_rotate,
                                cfg.aug_rotate),
                        shear=(-cfg.aug_shear,
                                cfg.aug_shear)),
                    iaa.SomeOf(1, [
                    iaa.AdditiveGaussianNoise(
                        scale=cfg.aug_noise*255),
                     iaa.GaussianBlur(sigma=(0., cfg.aug_blur)),
                     iaa.GammaContrast((0., cfg.aug_gamma))]),
                    iaa.Fliplr(p=0.5),
                    iaa.Flipud(p=0.5)]),
        rescale_augmenter])

    transf_normal = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dl_train = Loader(pjoin(cfg.in_root, 'Dataset'+cfg.train_dir),
                      augmentation=transf,
                      n_segments=cfg.n_segments_train,
                      delta_segments=cfg.delta_segments_train,
                      normalization=transf_normal)

    dl_test = torch.utils.data.ConcatDataset([Loader(pjoin(cfg.in_root,
                                                           'Dataset'+d),
                                                     augmentation=transf,
                                                     n_segments=cfg.n_segments_test,
                                                     delta_segments=cfg.delta_segments_test,
                                                     normalization=transf_normal)
                                              for d in cfg.test_dirs])

    dataloader_train = DataLoader(dl_train,
                                  batch_size=cfg.batch_size,
                                  sampler=SubsetRandomSampler(
                                      cfg.n_frames_epoch * cfg.train_frames),
                                  collate_fn=dl_train.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)

    dataloader_test = DataLoader(dl_test,
                                 batch_size=cfg.batch_size,
                                 collate_fn=dl_train.collate_fn,
                                 sampler=torch.utils.data.RandomSampler(
                                     dl_test,
                                     replacement=True,
                                     num_samples=cfg.batch_size),
                                 num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train,
                   'test': dataloader_test}

    d = datetime.datetime.now()

    ds_dir = os.path.split('Dataset'+cfg.train_dir)[-1]

    run_dir = pjoin(cfg.out_dir, '{}_{:%Y-%m-%d_%H-%M}_{}'.format(ds_dir, d,
                                                                  cfg.exp_name))

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

    optimizer = optim.SGD(params = [
        {'params': model.autoenc.encoder.parameters(), 'lr': cfg.lr_autoenc},
        {'params': model.autoenc.aspp.parameters(), 'lr': cfg.lr_autoenc},
        {'params': model.autoenc.decoder.parameters(), 'lr': cfg.lr_siam},
        {'params': model.linear1.parameters(), 'lr': cfg.lr_siam},
        {'params': model.linear2.parameters(), 'lr': cfg.lr_siam}
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
    p.add('--train-frames', nargs='+', type=int, required=True)
    p.add('--test-dirs', nargs='+', type=str, required=True)

    p.add('--checkpoint-autoenc', default=None)
    p.add('--checkpoint-siam', default=None)

    cfg = p.parse_args()

    main(cfg)
