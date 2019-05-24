#!/bin/bash

#SBATCH --mail-user=laurent.lejeune@artorg.unibe.ch
#SBATCH --job-name="dqn"
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --output=ksp.out

singularity exec ${HOME}/ksptrack-ubelix.simg zsh main.sh

