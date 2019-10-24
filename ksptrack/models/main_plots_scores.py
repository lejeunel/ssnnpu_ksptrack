import dirs
import torch
from unet_objprior import UNetObjPrior, predict
from dataloader import Dataloader
import os
import glob
import utils
from im_utils import one_channel_tensor_to_img, img_tensor_to_img
import utils as utls
import numpy as np
from skimage import (io, segmentation)
import matplotlib.pyplot as plt

params = {
    'cuda': False,
}

if (params['cuda']):
    device = 'gpu'
else:
    device = 'cpu'

in_shape = (400, 400)

dsets = [
    'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03'
]

exp_dir = 'pred_ksp'

for k, pred_dir in dirs.pred_dirs.items():
    if k not in dsets:
        break

    path_pred = os.path.join(dirs.root_dir,
                             'obj-priors',
                             exp_dir, dirs.pred_dirs[k])
    print(path_pred)
    ims = sorted(
        glob.glob(os.path.join(dirs.root_dir, k, 'input-frames', '*.png')))

    truths = sorted(
        glob.glob(
            os.path.join(dirs.root_dir, k, 'ground_truth-frames', '*.png')))

    truths_ksp = sorted(
        glob.glob(
            os.path.join(dirs.root_dir, k, 'results', dirs.res_dirs[k],
                         'results', '*.png')))

    cp_path = os.path.join(path_pred, 'checkpoint.pth.tar')

    unet_obj = UNetObjPrior(params)
    unet_obj.load_checkpoint(cp_path, device=device)
    unet_obj.model.eval()

    locs2d = utils.read_csv(
        os.path.join(dirs.root_dir, k, 'gaze-measurements', 'video1.csv'))
    dloader_manual = Dataloader(
        in_shape,
        im_paths=ims,
        truth_paths=truths,
        locs2d=locs2d,
        batch_size=1,
        cuda=params['cuda'])
    dloader_manual.mode = 'eval'

    dloader_ksp = Dataloader(
        in_shape,
        im_paths=ims,
        truth_paths=truths_ksp,
        locs2d=locs2d,
        batch_size=1,
        cuda=params['cuda'])
    dloader_ksp.mode = 'eval'

    path_out = os.path.join(path_pred, 'res', 'preds')
    if (not os.path.exists(path_out)):
        os.makedirs(path_out)

    preds = list()
    truths = list()
    f1s = list()
    auc = list()
    pr = list()
    rc = list()

    for i, (data_ksp, data_manual) in enumerate(
            zip(dloader_ksp, dloader_manual)):

        print('{}/{}'.format(i + 1, len(dloader_ksp)))
        pred = unet_obj.forward(data_ksp.image, data_ksp.obj_prior)
        pred = one_channel_tensor_to_img(pred)
        preds.append(pred)
        truth = img_tensor_to_img(data_manual.truth)
        truths.append(truth)
        img = img_tensor_to_img(data_manual.image)

        cont_truth = segmentation.find_boundaries(truth, mode='thick')
        idx_cont_truth = np.where(cont_truth)
        img[idx_cont_truth[0], idx_cont_truth[1], :] = (255, 0, 0)

        pred = utls.center_pred(pred)
        pred[idx_cont_truth[0], idx_cont_truth[1], :] = (255, 0, 0)

        all = np.concatenate((img, pred, truth), axis=1)
        io.imsave(os.path.join(path_out, 'pred_{:04d}.png'.format(i)), all)

    print('computing scores')
    truths = (np.asarray(truths) > 0).ravel()
    preds = np.asarray(preds).ravel()
    fpr, tpr, auc_, pr, rc, f1, _ = utls.get_all_scores(truths, preds, 1000)

    scores = {'f1': f1,
              'fpr': fpr,
              'tpr': tpr,
              'auc': auc,
              'pr': pr,
              'rc': rc}
    path_save_res = os.path.join(path_pred, 'scores.npz')
    np.savez(path_save_res, **scores)
