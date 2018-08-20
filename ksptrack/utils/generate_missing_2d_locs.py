import os
import numpy as np
import datetime
from labeling.cfgs import cfg_unet, cfg
from labeling.utils import my_utils as utls
from labeling.utils import learning_dataset
from labeling.utils import csv_utils as csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.morphology as morphology

extra_cfg = dict()

all_datasets = ['Dataset00']
ratio_missing = [.05, .1, .2, .4, .5]
n_sets = 5

csv_columns = ['frame', 'time', 'visible', 'x', 'y']

# Run KSP on all seqs with first gaze-set and make prediction experiment
for i in range(len(all_datasets)):
    set_confs = []
    extra_cfg['dataSetDir'] = all_datasets[i]
    extra_cfg['seq_type'] = cfg.datasetdir_to_type(extra_cfg['dataSetDir'])
    cfg_dict = cfg.cfg()
    cfg_dict.update(extra_cfg)
    conf = cfg.dict_to_munch(cfg_dict)

    print("dset: " + all_datasets[i])
    #for k in [1, 2, 3, 4, 5]:
    for n in range(n_sets):
        for k in [1]:
            for r in ratio_missing:
                save_out_csv = ('video{}_{}'
                                '_ratio_{}_n_{}.csv').format(k,
                                                        'missing',
                                                        int(100*r),
                                                        int(n))
                save_out_path = os.path.join(conf.dataInRoot,
                                            conf.dataSetDir,
                                            conf.gazeDir,
                                            save_out_csv)

                extra_cfg['csvFileName_fg'] = 'video' + str(k) + '.csv'

                orig_2dlocs = utls.readCsv(os.path.join(conf.dataInRoot,
                                                        conf.dataSetDir,
                                                        conf.gazeDir,
                                                        conf.csvFileName_fg))
                noised_2dlocs = orig_2dlocs.copy()

                conf.frameFileNames = utls.makeFrameFileNames(
                    conf.framePrefix, conf.frameDigits, conf.frameDir,
                    conf.dataInRoot, conf.dataSetDir, conf.frameExtension)


                n_frames = len(conf.frameFileNames)
                inds_to_noise = np.random.choice(np.arange(n_frames),
                                                size=int(n_frames*r),
                                                replace=False)

                noised_2dlocs = np.delete(noised_2dlocs,
                                        inds_to_noise,
                                        axis=0)
                # Save noised coordinates
                print('Saving to {}'.format(save_out_path))
                df = pd.DataFrame(data=noised_2dlocs,
                                columns=csv_columns)
                df.to_csv(path_or_buf=save_out_path)
