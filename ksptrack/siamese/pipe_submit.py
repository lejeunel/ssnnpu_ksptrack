import re
import os
import shutil
import fileinput
import subprocess

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

args = {
    'job_name': [job_prefix + n for n in job_names],
    'run_dirs': run_dirs,
    'train_dirs': train_dirs
}

template = 'mysubmit_tmpl.sh'
file_ = 'mysubmit_tmp.sh'

job_mask = [True, True, True, True, True, True]

for j in range(len(job_names)):
    shutil.copyfile(template, file_)
    if (job_mask[j]):
        for k, v in args.items():
            for line in fileinput.input(file_, inplace=True):
                line = re.sub('###{}###'.format(k), v[j], line.rstrip())
                print(line)

        print('-----------------------------------')
        print('starting job {}'.format(j))
        print('-----------------------------------')
        os.system('cat {}'.format(file_))
        print('-----------------------------------')
        subprocess.call(["sbatch", file_])
