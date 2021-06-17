import numpy as np
import pandas as pd

path = '/home/ubelix/lejeune/data/medical-labeling/Dataset00/precomp_desc/sp_desc_siam.p'

out = pd.read_pickle(path)

clicked = np.zeros(out.shape[0]).astype(bool)
fl = out[['frame', 'label']]
fl['positive'] = False
to_add = pd.DataFrame([(0, 2), (0, 3)], columns=['frame', 'label'])
to_add['positive'] = True
new = pd.merge(fl,
                to_add,
                how='left',
                on=['frame', 'label']).fillna(False)
new['positive'] = (new['positive_x'] + new['positive_y'])
new = new.drop(columns=['positive_x', 'positive_y'])
