#!/bin/env bash

#SBATCH --job-name=###job_name###
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/logs/%x.out

dir=$HOME/Documents/software/ksptrack/ksptrack/exps
simg=$HOME/ksptrack-ubelix.simg
pyversion=my-3.7
exec=python
script=pipe_trans.py

export OMP_NUM_THREADS=1

args="--cuda --out-path $HOME/runs/ksptrack --root-path $HOME --sets ###sets###"
# args="--out-path $HOME/runs/ksptrack --root-path $HOME --sets ###sets### --set-labeled ###set_labeled### --labeled-frames ###labeled_frames###"

singularity exec --nv $simg /bin/bash -c "source $HOME/.bashrc && pyenv activate $pyversion && cd $dir && $exec $script $args"
