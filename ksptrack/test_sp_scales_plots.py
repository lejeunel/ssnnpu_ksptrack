import os
import numpy as np
import datetime
from labeling.cfgs import cfg
import yaml
import numpy as np
import matplotlib.pyplot as plt
from labeling.utils import my_utils as utls
from labeling.utils.data_manager import DataManager
from labeling.utils import superpixel_extractor as svx
from sklearn.metrics import f1_score
import bunch

root_dir = '/home/laurent.lejeune/medical-labeling'
dirs = ['Dataset00/results/2018-04-20_12-29-04_sp_scales',
        'Dataset01/results/2018-04-20_12-57-11_sp_scales',
        'Dataset02/results/2018-04-20_13-26-35_sp_scales',
        'Dataset03/results/2018-04-20_13-54-03_sp_scales',
        'Dataset10/results/2018-04-20_12-36-44_sp_scales',
        'Dataset11/results/2018-04-20_12-52-57_sp_scales',
        'Dataset12/results/2018-04-20_13-06-43_sp_scales',
        'Dataset13/results/2018-04-20_13-25-48_sp_scales',
        'Dataset20/results/2018-04-20_12-36-35_sp_scales',
        'Dataset21/results/2018-04-20_12-52-09_sp_scales',
        'Dataset22/results/2018-04-20_13-04-58_sp_scales',
        'Dataset23/results/2018-04-20_13-12-52_sp_scales',
        'Dataset30/results/2018-04-20_12-32-24_sp_scales',
        'Dataset31/results/2018-04-20_12-41-26_sp_scales',
        'Dataset32/results/2018-04-20_12-47-58_sp_scales',
        'Dataset33/results/2018-04-20_12-55-05_sp_scales'
]

titles = ['00',
          '01',
          '02',
          '03',
          '10',
          '11',
          '12',
          '13',
          '20',
          '21',
          '22',
          '23',
          '30',
          '31',
          '32',
          '33']

types = {'0': 'Tweezer',
         '1': 'Cochlea',
         '2': 'Slitlamp',
         '3': 'Brain'}

sp_scales = [8000, 13000, 18000, 23000]

res = dict()

for d, t in zip(dirs, titles):
    f1s_ = []
    nsp_ = []
    for s in sp_scales:
        print('dir: {}\n scale: {}'.format(d, s))
        f_ = np.load(os.path.join(root_dir,
                                  d,
                                  'results_scale_{}.npz'.format(s)))
        labels = f_['labels']
        f1 = f_['f1']
        scale = f_['scale']

        nsp = np.unique(labels[..., 0]).size

        f1s_.append(f1)
        nsp_.append(nsp)

    res[t] = {'f1': f1s_,
              'nsp': nsp_,
              'scale': scale}

type_ = '0'
res_ = {k:v for k, v in res.items() if(k[0] == type_)}
for k in res_.keys():
    plt.plot(res_[k]['nsp'],
             res_[k]['f1'],
             'o-')
plt.title(types[type_])
plt.xlabel('Num. of SP per frame')
plt.ylabel('F1 score')
plt.show()
