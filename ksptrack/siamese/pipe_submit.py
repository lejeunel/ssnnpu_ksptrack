from string import Template
import subprocess
from os.path import join as pjoin

job_prefix = 'sm'
job_names = ['tw', 'co', 'sl', 'br', 'sp', 'lv']
seq_type = list(range(6))
n_seqs_per_type = [6, 4, 6, 6, 6, 5]

run_dirs = [
    ' '.join([
        'Dataset' + str(t) + str(n) for n in list(range((n_seqs_per_type[t])))
    ]) for t in seq_type
]
train_dirs = [
    ' '.join([str(t) + str(n) for n in list(range(n_seqs_per_type[t]))])
    for t in seq_type
]

script_path = '$HOME/Documents/software/ksptrack/ksptrack/siamese'
script_name = 'train_all_type.py'

out_root = pjoin('$HOME/runs', 'siamese')
flags = '--cuda --skip-train-dec'

# out_root = pjoin('$HOME/runs', 'siamese_dec')
# flags = '--cuda --skip-train-dec'

args = [{
    'job_name': job_prefix + jn,
    'run_dirs': rd,
    'train_dirs': td,
    'script_path': script_path,
    'script_name': script_name,
    'in_root': '$HOME/data/medical-labeling',
    'out_root': out_root,
    'flags': flags
} for jn, rd, td in zip(job_names, run_dirs, train_dirs)]

template = """#!/bin/env bash

#SBATCH --job-name=$job_name
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/logs/%x.out

simg=$$HOME/ksptrack-ubelix.simg
pyversion=my-3.7

export OMP_NUM_THREADS=1

args="$flags --out-root $out_root --in-root $in_root --train-dirs $train_dirs --run-dirs $run_dirs"

singularity exec --nv $$simg /bin/bash -c "source $$HOME/.bashrc && pyenv activate $$pyversion && cd $script_path && python $script_name $$args"

"""

job_mask = [True, True, True, True, True, True]

for i, args_ in enumerate(args):
    if (job_mask[i]):
        src = Template(template)
        content = src.substitute(args_)

        print('-----------------------------------')
        print(content)
        print('-----------------------------------')
        text_file = open('tmp_', 'w')
        text_file.write(content)
        text_file.close()
        subprocess.call(["sbatch", 'tmp_'])
