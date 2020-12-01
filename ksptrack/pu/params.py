import argparse
from os.path import join as pjoin

import configargparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_params(path='.'):
    p = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[pjoin(path, 'default.yaml')])

    p.add('-v', help='verbose', action='store_true')

    p.add('--data-type')
    p.add('--frames', action='append', type=int)
    p.add('--n-patches', type=int)
    p.add('--epochs-autoenc', type=int)
    p.add('--epochs-aug', type=int)
    p.add('--epochs-pred', type=int)
    p.add('--epochs-pre-pred', type=int)
    p.add('--epochs-post-pred', type=int)
    p.add('--epochs-dist', type=int)
    p.add('--epochs-dec', type=int)
    p.add('--prev-period', type=int)
    p.add('--prior-period', type=int)
    p.add('--cp-period', type=int)
    p.add('--unlabeled-period', type=int)
    p.add('--unlabeled-ratio', type=float)
    p.add('--tgt-update-period', type=int)
    p.add('--cc-update-period', type=int)
    p.add('--neg-mode', type=str)
    p.add('--loss-obj-pred', type=str)
    p.add('--aug-method', type=str)
    p.add('--pred-init-dir', type=str)
    p.add('--nnpu-ascent', default=False, action='store_true')
    p.add('--aug-in-neg', default=False, action='store_true')
    p.add('--aug-reset', default=False, action='store_true')
    p.add('--do-scores', default=False, action='store_true')
    p.add('--pxls', default=False, action='store_true')

    p.add('--phase', default=0, type=int)

    p.add('--n-frames-epoch', type=int)
    p.add('--momentum', type=float)
    p.add('--lambda-', type=float)
    p.add('--lr-dist', type=float)
    p.add('--lr-autoenc', type=float)
    p.add('--lr-assign', type=float)
    p.add('--lr-power', type=float)
    p.add('--lr-epoch-milestone-0', type=int)
    p.add('--lr-epoch-milestone-1', type=int)
    p.add('--lr-gamma', type=float)
    p.add('--decay', type=float)
    p.add('--beta1', type=float)
    p.add('--beta2', type=float)
    p.add('--beta', type=float)

    p.add('--lr0', type=float)
    p.add('--lr1', type=float)
    p.add('--lr2', type=float)
    p.add('--ds-split', type=float)
    p.add('--ds-shuffle', type=bool)
    p.add('--batch-size', type=int)
    p.add('--batch-norm', type=bool)
    p.add('--cuda', default=False, action='store_true')
    p.add('--in-shape', type=int)

    p.add('--n-ims-test', type=int)

    p.add('--aug-noise', type=float)
    p.add('--aug-scale', type=float)
    p.add('--aug-blur-color-low', type=int)
    p.add('--aug-blur-color-high', type=int)
    p.add('--aug-blur-space-low', type=int)
    p.add('--aug-blur-space-high', type=int)
    p.add('--aug-gamma-low', type=float)
    p.add('--aug-gamma-high', type=float)
    p.add('--aug-rotate', type=float)
    p.add('--aug-shear', type=float)
    p.add('--aug-flip-proba', type=float)
    p.add('--aug-some', type=int)

    p.add('--exp-name', type=str)

    p.add('--reduc-method')
    p.add('--ml-up-thr', type=float)
    p.add('--ml-down-thr', type=float)
    p.add('--bag-t', type=int)
    p.add('--bag-n-feats', type=float)
    p.add('--bag-max-depth', type=float)

    p.add('--pi-mul', type=float)
    p.add('--init-pi', type=float)
    p.add('--pi-xi', type=float)
    p.add('--pi-overspec-ratio', type=float)
    p.add('--pi-post-ratio', type=float)
    p.add('--pi-min', type=float)
    p.add('--gamma', type=float)
    p.add('--pi-alpha', type=float)
    p.add('--pi-eta', type=float)
    p.add('--var-thr', type=float)
    p.add('--var-epc', type=int)
    p.add('--min-var-epc', type=int)
    p.add('--rho-pi-err', type=float)
    p.add('--n-classes-otsu', type=int)

    p.add('--loc-prior', default=False, action='store_true')
    p.add('--pi-filt-size', type=float)
    p.add('--pi-filt', default=False, action='store_true')
    p.add('--true-prior', default=False, action='store_true')
    p.add('--coordconv', default=False, action='store_true')

    p.add('--trans-fac', type=float)
    p.add('--obs-fac', type=float)
    p.add('--xi-fac-start', type=float)
    p.add('--xi-fac-end', type=float)

    return p
