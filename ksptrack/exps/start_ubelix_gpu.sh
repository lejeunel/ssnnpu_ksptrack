#!/bin/bash

#SBATCH --mail-user=laurent.lejeune@artorg.unibe.ch
#SBATCH --job-name="ksp_tweez"
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=ksp_tweez.out

CMD="python pipe_circle_masks.py --cuda --out-path ${HOME}/runs/ksptrack --root-path ${HOME} --sets 00 01 02 03 04 05 --set-labeled 00 --labeled-frames 15"
DIR=$HOME/Document/software/ksptrack/ksptrack/exps
singularity exec --nv $HOME/ksptrack-ubelix.simg /bin/zsh -c "cd ${DIR} && ${CMD}"

