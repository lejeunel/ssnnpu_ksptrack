#!/bin/env bash

#SBATCH --job-name=###job_name###
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/logs/%x.out

dir=$HOME/Documents/software/siamese_sp/siamese_sp
simg=$HOME/ksptrack-ubelix.simg
pyversion=my-3.7
exec=python
script=train_all.py

export OMP_NUM_THREADS=1

args="--cuda --in-root $HOME/data/medical-labeling --out-dir $HOME/runs/siamese_dec --train-dir ###train_dirs### "

singularity exec --nv $simg /bin/bash -c "source $HOME/.bashrc && pyenv activate $pyversion && cd $dir && $exec $script $args"
