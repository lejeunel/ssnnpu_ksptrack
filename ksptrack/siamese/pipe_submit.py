from string import Template
import subprocess
from os.path import join as pjoin


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [
        alist[i * length // wanted_parts:(i + 1) * length // wanted_parts]
        for i in range(wanted_parts)
    ]


n_jobs = 4

seq_type = list(range(4))
# seq_type = list(range(2))
seqs_per_type = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
# seqs_per_type = [[0], [0], [0], [0]]

run_dirs = [['Dataset' + str(t) + str(n) for n in seqs_per_type[t]]
            for t in seq_type]

train_dirs = [[str(t) + str(n) for n in seqs_per_type[t]] for t in seq_type]

run_dirs = [item for sublist in run_dirs for item in sublist]

n_jobs = min((len(run_dirs), n_jobs))
run_dirs = split_list(run_dirs, n_jobs)

train_dirs = [item for sublist in train_dirs for item in sublist]
train_dirs = split_list(train_dirs, n_jobs)

script_path = '$HOME/Documents/software/ksptrack/ksptrack/siamese'
script_name = 'train_all_type.py'
# script_name = 'train_grid_focal.py'

out_root = pjoin('$HOME/runs', 'siamese_dec')
flags = '--cuda'

job_prefix = 'sm'

args = [{
    'job_name': job_prefix + str(i),
    'run_dirs': ' '.join(rd),
    'train_dirs': ' '.join(td),
    'script_path': script_path,
    'script_name': script_name,
    'in_root': '$HOME/data/medical-labeling',
    'out_root': out_root,
    'flags': flags
} for i, (rd, td) in enumerate(zip(run_dirs, train_dirs))]

template = """#!/bin/env bash

#SBATCH --job-name=$job_name
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/logs/%x.out

simg=$$HOME/mleval-ubelix.simg
pyversion=my-3.7

export OMP_NUM_THREADS=1

args="$flags --out-root $out_root --in-root $in_root --train-dirs $train_dirs --run-dirs $run_dirs"

singularity exec --nv $$simg /bin/bash -c "source $$HOME/.bashrc && conda activate my && cd $script_path && python $script_name $$args"

"""

# job_mask = [True, False, False, False, False, False]
# job_mask = [True, True, True, True, False, False]

for i, args_ in enumerate(args):
    # if (job_mask[i]):
    src = Template(template)
    content = src.substitute(args_)

    print('-----------------------------------')
    print(content)
    print('-----------------------------------')
    text_file = open('tmp_', 'w')
    text_file.write(content)
    text_file.close()
    subprocess.call(["sbatch", 'tmp_'])
