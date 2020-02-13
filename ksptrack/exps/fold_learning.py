from ksptrack.cfgs import params
from os.path import join as pjoin
import os
import itertools
import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from ksptrack.utils.base_dataset import BaseDataset
import tqdm
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.models.utils import save_checkpoint
import numpy as np
import torch.optim as optim
from skimage import io
from torch.nn import functional as F


def make_dataloaders(cfg, train_folds, test_fold,
                     batch_size):

    dls_train = []
    for train_fold in train_folds:
        dl = BaseDataset(train_fold['in_path'])
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
    dl_test = DataLoader(BaseDataset(test_fold['in_path']),
                         collate_fn=collate_fn,
                         batch_size=batch_size)
    dl_prev = DataLoader(BaseDataset(test_fold['in_path']),
                         collate_fn=collate_fn,
                         sampler=SubsetRandomSampler(
                             np.random.choice(len(dl_test), 5)),
                         batch_size=1)

    dataloaders = {'train': dl_train, 'val': dl_val, 'prev': dl_prev}

    return dataloaders


def train_one_fold(cfg, train_folds, test_fold, out_path,
                   batch_size):

    dataloaders = make_dataloaders(cfg, train_folds, test_fold,
                                   batch_size)

    test_im_dir = pjoin(out_path)
    if (not os.path.exists(test_im_dir)):
        os.makedirs(test_im_dir)

    criterion = torch.nn.MSELoss()

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    model = DeepLabv3Plus(pretrained=False, num_classes=1)
    model.to(device)

    optimizer = optim.SGD(params=[{
        'params': model.parameters(),
        'lr': 0.1
    }],
                          momentum=0.9,
                          weight_decay=0.00004)
    losses = []

    n_epochs = 100
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

                else:
                    import pdb; pdb.set_trace() ## DEBUG ##
                    prev_ims.update({
                        data['frame_name'][0]:
                        np.rollaxis(data['image'][0].cpu().detach().numpy(), 0,
                                    3)
                    })
                    prev_ims_pred.update({
                        data['frame_name'][0]:
                        np.rollaxis(output[0].cpu().detach().numpy(), 0,
                                    3)
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
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()
                    },
                    is_best,
                    fname_cp='checkpoint_pred.pth.tar',
                    fname_bm='best_pred.pth.tar',
                    path=path)
            else:
                # save previews
                import pdb; pdb.set_trace() ## DEBUG ##
                prev_ims = np.vstack(
                    [prev_ims[k] for k in sorted(prev_ims.keys())])
                prev_ims_recons = np.vstack(
                    [prev_ims_pred[k] for k in sorted(prev_ims_pred.keys())])
                all = np.concatenate((prev_ims, prev_ims_recons), axis=1)

                io.imsave(pjoin(test_im_dir, 'im_{:04d}.png'.format(epoch)),
                          all)


def main(cfg):
    folds = [{
        'in_path': in_path,
        'run_path': run_path
    } for in_path, run_path in zip(cfg.in_paths, cfg.run_paths)]
    for train_folds in itertools.combinations(folds, len(folds) - 1):
        print('training with {}'.format(train_folds))
        test_fold = [f for f in folds if (f not in train_folds)][0]
        print('testing with {}'.format(test_fold))
        train_one_fold(cfg, train_folds, test_fold, cfg.out_path,
                       2)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-path', required=True)
    # p.add('--in-dirs', nargs='+', required=True)
    p.add('--in-root', required=True)
    p.add('--run-root', required=True)
    # p.add('--run-paths', nargs='+', required=True)

    cfg = p.parse_args()

    cfg.run_dirs = [
        'Dataset20/transexp_up_0.60_down_0.20',
        'Dataset21/transexp_up_0.60_down_0.20',
        'Dataset22/transexp_up_0.60_down_0.20',
        'Dataset23/transexp_up_0.60_down_0.20',
        'Dataset24/transexp_up_0.60_down_0.20'
    ]

    cfg.in_dirs = [
        'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24'
    ]

    cfg.in_paths = [pjoin(cfg.in_root, d) for d in cfg.in_dirs]
    cfg.run_paths = [pjoin(cfg.run_root, d) for d in cfg.run_dirs]

    main(cfg)
