import params
from ksptrack.siamese import train_autoencoder, train_dec, train_init_clst, train_siam, train_focal


def main(cfg):
    train_autoencoder.main(cfg)
    train_init_clst.main(cfg)

    if (cfg.siamese != 'none'):
        train_siam.main(cfg)
    else:
        train_focal.main(cfg)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)
    p.add('--init-cp-fname')

    cfg = p.parse_args()
    main(cfg)
