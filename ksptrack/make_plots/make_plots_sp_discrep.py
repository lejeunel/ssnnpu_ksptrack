import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labeling.exps import results_dirs as rd
import shutil as sh

path_ = os.path.join(rd.root_dir, 'plots_results', 'sp_thr_f1s.npz')

data = np.load(path_)
res = data['res'][()]
ratios = data['ratios']

for k in res.keys():
    f1s = np.asarray([res[k][dset] for dset in res[k].keys()])

    plt.errorbar(ratios, f1s.mean(axis=0), yerr=f1s.std(axis=0),
                fmt='--',
                capsize=5,
                elinewidth=2,
                markeredgewidth=2)

plt.grid()
plt.xlabel('Overlap ratio[%]')
plt.ylabel('F1')
plt.show()
