#!/bin/zsh

ROOT_DIR=$HOME/medical-labeling
MAIN_DIR=$HOME/Documents/software/ksptrack/ksptrack

source ~/.zshrc
python --version
pyenv versions

cd $MAIN_DIR

# python main.py -c env.yaml --out-dir $ROOT_DIR/dqn/tweezer --env-root-dir $ROOT_DIR
python single_ksp_gpu.py --in-path $ROOT_DIR/medical-labeling/Dataset00 --out-path $ROOT_DIR/Dataset00 --entrance-masks-path $ROOT_DIR/unet_region/runs/Dataset00_2019-04-10_11-38-11/Dataset00/entrance_masks --bag-t 500 --bag-jobs 4

