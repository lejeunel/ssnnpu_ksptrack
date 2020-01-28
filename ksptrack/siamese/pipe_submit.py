import re
import os
import shutil
import fileinput
import subprocess

args = {
    'job_name':
    ['siam_tw', 'siam_co', 'siam_sl', 'siam_br', 'siam_sp', 'siam_lv'],
    'train_dir': [
        '00', '10', '20',
        '30', '40', '50'],
    'run_dir': [
        'Dataset00',
        'Dataset10',
        'Dataset20',
        'Dataset30',
        'Dataset40',
        'Dataset50']
}

template = 'mysubmit_tmpl.sh'
file_ = 'mysubmit_tmp.sh'

job_mask = [True, True, True, True, False, False]

for j in range(len(args['job_name'])):
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
