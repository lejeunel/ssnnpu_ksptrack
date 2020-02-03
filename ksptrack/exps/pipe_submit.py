import re
import os
import shutil
import fileinput
import subprocess

job_prefix = 'trans_'
args = {
    'job_name': [job_prefix+j for j in ['tw', 'co', 'sl', 'br', 'sp', 'lv']],
    'sets': [
        '00 01 02 03 04 05', '10 11 12 13', '20 21 22 23 24 25',
        '30 31 32 33 34 35', '40 41 42 43 44 45', '50 51 52 53 54'
    ],
    'set_labeled': ['00', '10', '20', '30', '40', '50'],
    'labeled_frames': ['15', '52', '15', '52', '102', '59']
}


job_mask = [True, False, False, False, False, False]
n_jobs = len(job_mask)
template = 'mysubmit_tmpl.sh'
file_ = 'mysubmit_tmp.sh'

for j in range(n_jobs):
    if (job_mask[j]):
        shutil.copyfile(template, file_)
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
