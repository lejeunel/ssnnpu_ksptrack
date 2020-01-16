import logging
import torch.optim as optim
import params
import torch
import os
from os.path import join as pjoin
import yaml
import utils as utls
from siamese_sp import train_autoencoder, train_dec


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dirs', nargs='+', required=True)

    cfg = p.parse_args()

    for d in cfg.train_dirs:
        cfg.train_dir = d
        train_autoencoder.main(cfg)
        train_dec.main(cfg)
