import params
from ksptrack.pu import train_autoencoder, train_obj_pred
from os.path import join as pjoin
import os


def main(cfg):
    # train_autoencoder.main(cfg)

    check_dir_exist = pjoin(cfg.out_root, cfg.run_dir, cfg.exp_name)
    if os.path.exists(check_dir_exist):
        print('path {} already exists, skipping.'.format(check_dir_exist))
    else:
        train_obj_pred.main(cfg)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)
    p.add('--init-cp-fname')

    cfg = p.parse_args()

    main(cfg)
