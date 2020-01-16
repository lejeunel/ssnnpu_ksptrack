#!/bin/env bash

#SBATCH --job-name=siam_br
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=all
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/logs/%x.out

dir=$HOME/Documents/siamese_sp/siamese_sp
simg=$HOME/ksptrack-ubelix.simg
pyversion=my-3.7
exec=python
script=train.py

export OMP_NUM_THREADS=1

args="--cuda --in-root $HOME/data/medical-labeling --out-dir $HOME/runs/siamese --train-dir 30 --train-frames 52 --test-dirs 30 31 32 33 34 35 --sp-pooling-max "

singularity exec --nv $simg /bin/bash -c "source $HOME/.bashrc && pyenv activate $pyversion && cd $dir && $exec $script $args"
