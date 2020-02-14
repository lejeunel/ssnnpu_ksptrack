import configargparse
import argparse
from os.path import join as pjoin


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
    p.add('--epochs-dist', type=int)
    p.add('--epochs-dec', type=int)
    p.add('--prev-period', type=int)
    p.add('--cp-period', type=int)
    p.add('--tgt-update-period', type=int)

    p.add('--n-frames-epoch', type=int)
    p.add('--momentum', type=float)
    p.add('--lr-dist', type=float)
    p.add('--lr-autoenc', type=float)
    p.add('--lr-assign', type=float)
    p.add('--lr-power', type=float)
    p.add('--decay', type=float)

    p.add('--lr', type=float)
    p.add('--ds-split', type=float)
    p.add('--ds-shuffle', type=bool)
    p.add('--batch-size', type=int)
    p.add('--batch-norm', type=bool)
    p.add('--n-workers', type=int)
    p.add('--cuda', default=False, action='store_true')
    p.add('--in-shape', type=int)

    p.add('--n-ims-test', type=int)

    p.add('--n-segments-test', type=int)
    p.add('--delta-segments-test', type=int)
    p.add('--n-segments-train', type=int)
    p.add('--delta-segments-train', type=int)

    p.add('--aug-noise', type=float)
    p.add('--aug-scale', type=float)
    p.add('--aug-blur', type=float)
    p.add('--aug-gamma', type=float)
    p.add('--aug-rotate', type=float)
    p.add('--aug-shear', type=float)
    p.add('--aug-flip-proba', type=float)
    p.add('--aug-some', type=int)

    p.add('--sp-pooling-max', default=False, action='store_true')

    p.add('--exp-name', default='')
    p.add('--n-edges-ratio', type=float)
    p.add('--nn-radius', type=float)
    p.add('--n-clusters', type=int)
    p.add('--embedded-dims', type=int)

    p.add('--roi-output-size', type=int)
    p.add('--roi-spatial-scale', type=float)

    p.add('--with-pck', default=False, action='store_true')
    p.add('--with-agglo', default=False, action='store_true')
    p.add('--with-flow', default=False, action='store_true')
    p.add('--reduc-method')
    p.add('--ml-up-thr', type=float)
    p.add('--ml-down-thr', type=float)
    p.add('--bag-t', type=int)

    p.add('--linkage', type=str)

    p.add('--beta', type=float)
    p.add('--gamma', type=float)
    p.add('--alpha', type=float)
    p.add('--lambda-', type=float)

    p.add('--thr-assign', type=float)
    p.add('--skip-train-dec', default=False, action='store_true')
    return p
