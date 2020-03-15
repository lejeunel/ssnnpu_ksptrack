from ksptrack.siamese import params
from ksptrack.siamese import train_all
from ksptrack import iterative_ksp
from ksptrack.cfgs import params as params_ksp
from os.path import join as pjoin
import os


if __name__ == "__main__":

    p = params.get_params()
    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dirs', nargs='+', required=True)
    p.add('--run-dirs', nargs='+', required=True)
    cfg = p.parse_args()

    p_ksp = params_ksp.get_params('../cfgs')
    p_ksp.add('--out-path')
    p_ksp.add('--in-path')
    p_ksp.add('--siam-path', default='')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]

    for run_dir, train_dir in zip(cfg.run_dirs, cfg.train_dirs):
        cfg.run_dir = run_dir
        cfg.train_dir = train_dir
        cfg_ksp.out_path = pjoin(os.path.split(cfg.out_root)[0],
                                 'ksptrack', cfg.run_dir)
        cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
        cfg_ksp.cuda = True

        cfg.pw = False
        train_all.main(cfg)

        # run without DML
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir,
                                  'checkpoints',
                                  'init_dec.pth.tar')
        cfg_ksp.exp_name = 'exp_lfda'
        iterative_ksp.main(cfg_ksp)

        # run with DML
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir,
                                  'checkpoints',
                                  'checkpoint_siam.pth.tar')
        cfg_ksp.exp_name = 'exp_dml'
        iterative_ksp.main(cfg_ksp)

        # train DML + PW constraints
        # cfg.pw = True
        # train_all.main(cfg)

        # cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir,
        #                           'checkpoints',
        #                           'checkpoint_siam_pw.pth.tar')
        # cfg_ksp.exp_name = 'exp_dml_pw'
        # iterative_ksp.main(cfg_ksp)


