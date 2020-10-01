#!/bin/env bash

#SBATCH --job-name=siam_co
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/logs/%x-%j.out

dir=$HOME/Documents/software/siamese_sp/siamese_sp
simg=$HOME/ksptrack-ubelix.simg
pyversion=my-3.7
exec=python
script=train.py

# args="--cuda --in-root $HOME/data/medical-labeling --out-dir $HOME/runs/siamese --train-dir 00 --train-frames 15 --test-dirs 00 01 02 03 04 05 --sp-pooling-max --exp-name maxpool"
args="--cuda --in-root $HOME/data/medical-labeling --out-dir $HOME/runs/siamese --train-dir 10 --train-frames 51 --test-dirs 10 11 12 13 --sp-pooling-max --exp-name maxpool"
# args="--cuda --in-root $HOME/data/medical-labeling --out-dir $HOME/runs/siamese --train-dir 20 --train-frames 15 --test-dirs 20 21 22 23 24 25 --sp-pooling-max --exp-name maxpool"
# args="--cuda --in-root $HOME/data/medical-labeling --out-dir $HOME/runs/siamese --train-dir 30 --train-frames 52 --test-dirs 30 31 32 33 34 35"
# args="--cuda --in-root $HOME/data/medical-labeling --out-dir $HOME/runs/siamese --train-dir 30 --train-frames 52 --test-dirs 30 31 32 33 34 35 --sp-pooling-max --exp-name maxpool"

export OMP_NUM_THREADS=1

## use ${dir}, ${simg}, ${pyversion}, ${exec}, ${script}, ${args} below
singularity exec --nv $simg /bin/bash -c "source $HOME/.bashrc && pyenv activate $pyversion && cd $dir && $exec $script $args"

# singularity exec --nv $simg /bin/bash -c "source $HOME/.bashrc && echo $LD_LIBRARY_PATH"
