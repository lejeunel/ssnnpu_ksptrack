from string import Template
import subprocess
from os.path import join as pjoin

job_prefix = 'tr'
job_names = ['tw', 'co', 'sl', 'br', 'sp', 'lv']
seq_type = list(range(6))
n_seqs_per_type = [6, 4, 6, 6, 6, 5]

sets = [
    ' '.join([
        str(t) + str(n) for n in list(range((n_seqs_per_type[t])))
    ]) for t in seq_type
]

script_path = '$HOME/Documents/software/ksptrack/ksptrack/exps'
script_name = 'pipe_trans.py'
out_path = pjoin('$HOME/runs', 'ksptrack')
# siam_run_root = pjoin('$HOME/runs', 'siamese_dec')
siam_run_root = '\'\''
flags = '--cuda'

args = [{
    'job_name': job_prefix + jn,
    'sets': sets_,
    'script_path': script_path,
    'script_name': script_name,
    'siam_run_root': siam_run_root,
    'root_path': '$HOME',
    'out_path': out_path,
    'flags': flags
} for jn, sets_ in zip(job_names, sets)]

template = """#!/bin/env bash

#SBATCH --job-name=$job_name
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --output=/home/ubelix/artorg/lejeune/runs/logs/%x.out

simg=$$HOME/ksptrack-ubelix.simg
pyversion=my-3.7

export OMP_NUM_THREADS=1

args="$flags --out-path $out_path --root-path $root_path --sets $sets --siam-run-root $siam_run_root"

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
