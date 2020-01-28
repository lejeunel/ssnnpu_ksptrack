import logging
import torch.optim as optim
import params
import torch
import os
from os.path import join as pjoin
import yaml
import utils as utls
from ksptrack.siamese import train_autoencoder, train_dec


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    cfg.lr_autoenc = 0.1
    train_autoencoder.main(cfg)

    cfg.lr_autoenc = 0.001
    train_dec.main(cfg)
