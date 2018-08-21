import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ksptrack.exps import results_dirs as rd
import shutil as sh
import pandas as pd

root_dir = os.path.join(rd.root_dir, 'plots_results')
path_ = os.path.join(root_dir, 'sp_thr_f1s.npz')

data = np.load(path_)
res = data['res'][()]
ratios = data['ratios']

all_f1_means = []

for k in res.keys():
    f1s = np.asarray([res[k][dset] for dset in res[k].keys()])
    all_f1_means.append(f1s.mean(axis=0))

    plt.plot(ratios, f1s.mean(axis=0), 'o-',
             label=k)

seq_types = res.keys()

plt.grid()
plt.gca().set_ylim(top=1.0)
plt.xticks(ratios)
plt.legend()
plt.xlabel('Overlap ratio[%]')
plt.ylabel('F1')
plt.show()
#plt.savefig(os.path.join(root_dir, 'sp_discrep.png'))

f1_df = pd.DataFrame(all_f1_means, ratios)
f1_df.index = list(seq_types)
f1_df.to_csv(path_or_buf=os.path.join(root_dir, 'sp_discrep.csv'))
