import params
from ksptrack.pu import train_autoencoder, train_obj_pred


def main(cfg):
    train_autoencoder.main(cfg)

    train_obj_pred.main(cfg)

    # if (cfg.siamese != 'none'):
    #     train_siam.main(cfg)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)
    p.add('--init-cp-fname')

    cfg = p.parse_args()
    main(cfg)
