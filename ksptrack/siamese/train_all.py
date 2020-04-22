import params
from ksptrack.siamese import train_autoencoder, train_dec, train_init_clst, train_siam


def main(cfg):
    train_autoencoder.main(cfg)

    # Train objectness predictor
    cfg.clf = True
    cfg.clf_reg = False
    cfg.pw = False
    cfg.epochs_dist = 20
    cfg.exp_name = 'pred'
    cfg.init_cp_fname = 'cp_pred.pth.tar'
    train_siam.main(cfg)

    train_init_clst.main(cfg)

    # train_dec.main(cfg)

    train_siam.main(cfg)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)
    p.add('--init-cp-fname')

    cfg = p.parse_args()
    main(cfg)
