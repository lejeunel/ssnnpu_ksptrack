#!/bin/env bash

#SBATCH --job-name=ksp_co
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/%x.out

dir=$HOME/Documents/software/ksptrack/ksptrack/exps
simg=$HOME/ksptrack-ubelix.simg
pyversion=my-3.7
exec=python
script=pipe_circle_masks.py

export OMP_NUM_THREADS=1

args="--cuda --out-path $HOME/runs/ksptrack --root-path $HOME --sets 10 11 12 13 14 15 --set-labeled 10 --labeled-frames 52"

singularity exec --nv $simg /bin/bash -c "source $HOME/.bashrc && pyenv activate $pyversion && cd $dir && $exec $script $args"
