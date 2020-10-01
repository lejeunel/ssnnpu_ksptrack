import params
from os.path import join as pjoin
import os
import itertools
import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from ksptrack.utils.base_dataset import BaseDataset
import tqdm
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.siamese import utils as utls
import numpy as np
import torch.optim as optim
from skimage import io
from torch.nn import functional as F
from ksptrack.siamese import im_utils
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import pandas as pd


def make_dataloaders(cfg, train_folds, test_fold, batch_size):

    _, transf_normal = im_utils.make_data_aug(cfg)

    dls_train = []
    for train_fold in train_folds:
        dl = BaseDataset(train_fold['in_path'], normalization=transf_normal)
        if(train_fold['run_path'] is not None):
            segs = np.load(pjoin(train_fold['run_path'],
                                'results.npz'))['ksp_scores_mat']
            dl.truths = [segs[..., i] for i in range(segs.shape[-1])]
        dls_train.append(dl)

    dl_train_val = ConcatDataset(dls_train)

    # keep 5% of training frames as validation
    idx_train_val = np.random.permutation(len(dl_train_val))
    idx_train = idx_train_val[:int(len(idx_train_val) * 0.95)]
    idx_val = [i for i in idx_train_val if (i not in idx_train)]

    collate_fn = dls_train[0].collate_fn
    dl_train = DataLoader(dl_train_val,
                          batch_size=batch_size,
                          sampler=SubsetRandomSampler(idx_train),
                          collate_fn=collate_fn,
                          drop_last=True)
    dl_val = DataLoader(dl_train_val,
                        batch_size=batch_size,
                        sampler=SubsetRandomSampler(idx_val),
                        collate_fn=collate_fn,
                        drop_last=True)
    dl_test = DataLoader(BaseDataset(test_fold['in_path'],
                                     normalization=transf_normal),
                         collate_fn=collate_fn)
    dl_prev = DataLoader(BaseDataset(test_fold['in_path']),
                         collate_fn=collate_fn,
                         sampler=SubsetRandomSampler(
                             np.random.choice(len(dl_test), 5)),
                         batch_size=1)

    dataloaders = {'train': dl_train, 'val': dl_val,
                   'test': dl_test,
                   'prev': dl_prev}

    return dataloaders


def train_one_fold(cfg, dataloaders, out_path, cp_fname,
                   best_cp_fname,
                   batch_size):


    test_im_dir = pjoin(out_path, 'prevs')
    if (not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)

    criterion = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    model = DeepLabv3Plus(pretrained=False,
                          do_skip=True,
                          num_classes=1)
    model.to(device)

    optimizer = optim.Adam(params=[{
        'params': model.parameters(),
        'lr': 0.001
    }],
                          weight_decay=0.00004)
    losses = []

    n_epochs = 20
    best_loss = float('inf')
    for epoch in range(n_epochs):
        for phase in dataloaders.keys():

            running_loss = 0.0

            prev_ims = {}
            prev_ims_pred = {}
            # Iterate over data.
            pbar = tqdm.tqdm(total=len(dataloaders[phase]))
            for i, data in enumerate(dataloaders[phase]):
                if (phase == 'train'):
                    model.train()
                else:
                    model.eval()
                data = batch_to_device(data)

                with torch.set_grad_enabled(phase == 'train'):
                    res = model(data['image'])
                    output = F.sigmoid(res['output'])

                if ((phase == 'train') or (phase == 'val')):
                    loss = criterion(output, data['label/segmentation'])

                    if (phase == 'train'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.cpu().detach().numpy()
                    loss_ = running_loss / ((i + 1) * batch_size)

                if(phase == 'val'):
                    prev_ims.update({
                        data['frame_name'][0]:
                        np.rollaxis(((data['image'] + 1) / 2)[0].cpu().detach().numpy(), 0,
                                    3)
                    })
                    prev_ims_pred.update({
                        data['frame_name'][0]:
                        np.rollaxis(output[0].cpu().detach().numpy(), 0, 3)
                    })

                pbar.set_description('[{}] epch {}/{} lss: {:.6f}'.format(
                    phase, epoch + 1, n_epochs, loss_))
                losses.append((phase, loss_))

                pbar.update(1)

            pbar.close()
            if ((phase == 'val')):
                # save checkpoint
                is_best = False
                if (loss_ < best_loss):
                    is_best = True
                    best_loss = loss_
                path = pjoin(out_path, 'checkpoints')
                utls.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()
                    },
                    is_best,
                    fname_cp=cp_fname,
                    fname_bm=best_cp_fname,
                    path=path)

                # save previews
                prev_ims = np.vstack(
                    [prev_ims[k] for k in sorted(prev_ims.keys())])
                prev_ims_pred = np.vstack(
                    [prev_ims_pred[k] for k in sorted(prev_ims_pred.keys())])
                prev_ims_pred = np.repeat(prev_ims_pred, 3, axis=-1)
                all = np.concatenate((prev_ims, prev_ims_pred), axis=1)

                io.imsave(pjoin(test_im_dir, 'im_{:04d}.png'.format(epoch)),
                          all)

def eval(cfg, dataloaders, out_path,
         best_cp_fname, res_path):

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    model = DeepLabv3Plus(pretrained=False,
                          do_skip=True,
                          num_classes=1)

    cp_path = pjoin(out_path, 'checkpoints', best_cp_fname)

    if (os.path.exists(cp_path)):
        print('loading checkpoint {}'.format(cp_path))
        state_dict = torch.load(cp_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(cp_path))
        return

    model.to(device)
    model.eval()

    outputs = []
    truths = []

    # Iterate over data.
    pbar = tqdm.tqdm(total=len(dataloaders['test']))
    for i, data in enumerate(dataloaders['test']):
        data = batch_to_device(data)

        with torch.no_grad():
            res = model(data['image'])

        output = F.sigmoid(res['output'])
        outputs.append(output.squeeze().cpu().numpy())
        truths.append(data['label/segmentation'].squeeze().cpu().numpy().astype(bool))
        pbar.update(1)
    pbar.close()

    truths = np.array(truths).ravel()
    outputs = np.array(outputs).ravel()
    fpr, tpr, _ = roc_curve(truths, outputs)
    precision, recall, _ = precision_recall_curve(truths,
                                                  outputs)
    f1 = (2 * (precision * recall) / (precision + recall)).max()
    data = {'f1': f1,
            'fpr': fpr[1],
            'tpr': tpr[1],
            'pr': precision[1],
            'rc': recall[1]}

    df = pd.Series(data)

    path_ = pjoin(out_path, 'scores.csv')
    print('saving scores to {}'.format(path_))
    df.to_csv(path_)


def main(cfg):
    folds = [{
        'in_path': in_path,
        'run_path': run_path
    } for in_path, run_path in zip(cfg.in_paths, cfg.run_paths)]

    for i, train_folds in enumerate(itertools.combinations(folds, len(folds) - 1)):
        if(cfg.folds is None):
            pass
        elif(i not in cfg.folds):
            pass
        else:
            print('training with {}'.format(train_folds))
            test_fold = [f for f in folds if (f not in train_folds)][0]
            print('testing with {}'.format(test_fold))
            cp_fname = 'checkpoint_fold_{}.pth.tar'.format(i)
            best_cp_fname = 'best_checkpoint_fold_{}.pth.tar'.format(i)
            out_path = pjoin(cfg.out_path, cfg.exp_dir, 'fold_{}'.format(i))
            check_cp = pjoin(out_path, 'checkpoints', best_cp_fname)

            dataloaders = make_dataloaders(cfg, train_folds, test_fold, 2)

            if(os.path.exists(check_cp)):
                print('found {} skipping fold'.format(check_cp))
            else:
                train_one_fold(cfg, dataloaders, out_path,
                            cp_fname,
                            best_cp_fname, 2)

            res_path = pjoin(out_path, 'scores_csv')
            if (os.path.exists(res_path)):
                print(
                    'results file {} found. skipping'.format(res_path))
                return
            else:
                eval(cfg, dataloaders, out_path,
                    best_cp_fname, res_path)



if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-path',
          required=True,
          help='path where results will be stored')
    p.add('--in-root',
          required=True,
          help='path of image data. Omit dataset directory')
    p.add('--run-root',
          help='path of segmentation data. Omit dataset directory')
    p.add('--exp-dir-seg',
          help='dir name where segmentations can be found')
    p.add('--exp-dir',
          required=True,
          help='dir name to save results (e.g tweezer)')
    p.add('--in-dirs',
          nargs='+',
          required=True,
          help='input directory (e.g Dataset00 Dataset01...)')
    p.add('--fold',
          nargs='+',
          required=False,
          help='set of folds to perform. Default: Do all')

    cfg = p.parse_args()

    cfg.in_paths = [pjoin(cfg.in_root, d) for d in cfg.in_dirs]
    if(cfg.run_root):
        cfg.run_paths = [pjoin(cfg.run_root, d, cfg.exp_dir_seg) for d in cfg.in_dirs]
    else:
        cfg.run_paths = len(cfg.in_dirs) * [None]

    main(cfg)
