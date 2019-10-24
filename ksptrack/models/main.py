import os
import glob
import numpy as np
from skimage import io
from unet_objprior import UNetObjPrior
from dataloader import Dataloader
from dataloader import MetaDataloader
import utils
"""
Train CNN using as inputs (image, object prior) to predict segmentation mask
"""

root_dir = os.path.join('/home', 'laurent.lejeune', 'medical-labeling')
frame_dir = 'input-frames'
locs2d_dir = 'gaze-measurements'

out_dir_prefix = 'tweezer'
dset_dirs = ['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03']
truth_dirs = [
    '2018-06-01_17-37-00_exp', '2017-12-06_17-56-51_exp',
    '2018-09-27_18-05-07_exp', '2017-12-07_02-50-27_exp'
]

# out_dir_prefix = 'brain'
# dset_dirs = ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33']
# truth_dirs = [
#     '2017-12-06_13-37-18_exp', '2017-12-06_14-40-17_exp',
#     '2017-12-06_19-41-02_exp', '2017-12-07_15-32-41_exp'
# ]

# out_dir_prefix = 'slitlamp'
# dset_dirs = ['Dataset20', 'Dataset21', 'Dataset22', 'Dataset23']
# truth_dirs = [
#     '2017-12-06_13-36-56_exp', '2017-12-06_17-07-27_exp',
#     '2017-12-06_18-46-06_exp', '2017-12-06_19-48-47_exp'
# ]

# out_dir_prefix = 'cochlea'
# dset_dirs = ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13']
# truth_dirs = [
#     '2017-12-06_14-26-17_exp', '2017-12-06_16-16-27_exp',
#     '2017-12-06_17-28-07_exp', '2017-12-06_20-50-10_exp'
# ]

out_dir = os.path.join(root_dir,
                       'obj-priors',
                       'pred_manual',
                       out_dir_prefix)

params = {
    'lr': 10e-4,
    'momentum': 0.9,
    'weight_decay': 0,
    'batch_size': 4,
    'num_epochs': 100,
    'cuda': True,
    'out_dir': out_dir
}

in_shape = (400, 400)
dls = list()

for i, d in enumerate(dset_dirs):
    im_paths = sorted(glob.glob(os.path.join(root_dir, d, frame_dir, '*.png')))

    # KSP truths
    # truth_paths = utils.get_truth_frames(
    #     os.path.join(root_dir, d, 'results', truth_dirs[i]))

    # Manual truths
    truth_paths = sorted(glob.glob(
        os.path.join(root_dir, d, 'ground_truth-frames', '*.png')))

    locs2d = utils.read_csv(os.path.join(root_dir, d, locs2d_dir, 'video1.csv'))
    dls.append(
        Dataloader(
            in_shape,
            im_paths,
            truth_paths=truth_paths,
            locs2d=locs2d,
            batch_size=params['batch_size'],
            cuda=params['cuda']))

for k in range(4):
    params['out_dir'] += '_{}'.format(k)
    print('Fold :{}'.format(k))
    print('out_dir: {}'.format(params['out_dir']))
    train_dl = MetaDataloader([d for i, d in enumerate(dls) if (i != k)],
                              'train')
    val_dl = MetaDataloader([d for i, d in enumerate(dls) if (i == k)], 'val')
    model = UNetObjPrior(params)
    model.train(train_dl, val_dl)
