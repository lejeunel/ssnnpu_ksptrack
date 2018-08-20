from ruamel import yaml
from labeling.cfgs import cfg
import os
import json
import glob
import shutil
import string
import subprocess

paths = [
    '/home/laurent.lejeune/medical-labeling/Dataset00',
    '/home/laurent.lejeune/medical-labeling/Dataset01',
    '/home/laurent.lejeune/medical-labeling/Dataset02',
    '/home/laurent.lejeune/medical-labeling/Dataset03',
    '/home/laurent.lejeune/medical-labeling/Dataset11',
    '/home/laurent.lejeune/medical-labeling/Dataset10',
    '/home/laurent.lejeune/medical-labeling/Dataset12',
    '/home/laurent.lejeune/medical-labeling/Dataset13',
    '/home/laurent.lejeune/medical-labeling/Dataset20',
    '/home/laurent.lejeune/medical-labeling/Dataset21',
    '/home/laurent.lejeune/medical-labeling/Dataset22',
    '/home/laurent.lejeune/medical-labeling/Dataset23',
    '/home/laurent.lejeune/medical-labeling/Dataset30',
    '/home/laurent.lejeune/medical-labeling/Dataset31',
    '/home/laurent.lejeune/medical-labeling/Dataset32',
    '/home/laurent.lejeune/medical-labeling/Dataset33'
]

for p in paths:
    p_ = glob.glob(os.path.join(p, 'results', '*', 'cfg.yml'))

    for c in p_:

        if(os.path.exists(c)):
            c_new = c[0:-4] + '_old.yml'

            # Save old config
            shutil.copy(c, c_new)

            #Replace header
            with open(c, "r") as myfile:
                data=myfile.read().replace('\n', '')

            if('all_gaze' in data):
                cmd = 'sh /home/krakapwa/bin/del_lines {} {} {}'.format(c, 13, 74)
                subprocess.call(cmd, shell=True)

            if(data[0] == '!'):
                print(c)

                data_new = data.replace("!!python/object:cfg.Bunch",
                                        "!!python/object:labeling.cfgs.cfg.Bunch")

                # get original dict
                #with open(c, 'r') as infile:
                cfg_dict = yaml.load(data_new).__dict__


                # Remove gaze locations
                cfg_dict.pop('myGaze_fg', None)
                cfg_dict.pop('all_gaze', None)

                with open(c, 'w') as yaml_file:
                    yaml.dump(cfg_dict, stream=yaml_file, default_flow_style=False)
