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
from skimage import io
cmap = plt.get_cmap('viridis')

extra_cfg = dict()

extra_cfg['calc_superpix'] = False  # Centroids and contours

all_datasets = ['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03']
#all_datasets = ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33']
#all_datasets = ['Dataset20', 'Dataset21', 'Dataset22', 'Dataset23']
#all_datasets = ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13']

# for docker image
extra_cfg['root_path'] = '/home/laurent.lejeune/medical-labeling/'
extra_cfg['dataOutRoot'] = '/home/laurent.lejeune/medical-labeling/'

for d in all_datasets:
    extra_cfg['ds_dir'] = d
    print(d)

    conf = bunch.Bunch(cfg.cfg())
    conf.update(extra_cfg)

    conf.out_dir_prefix = 'sp_scales'

    conf.frameFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.frameDigits, conf.frameDir,
        conf.root_path, conf.ds_dir, conf.frame_extension)

    conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot,
                                            conf.ds_dir,
                                            conf.resultDir,
                                            conf.out_dir_prefix,
                                            conf.testing)

    my_dataset = DataManager(conf)
    sp_scales = [8000, 13000, 18000, 23000]
    f1s = []
    labels_contours = []
    spix_extr = svx.SuperpixelExtractor()

    gtFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.frameDigits, conf.truth_dir,
        conf.root_path, conf.ds_dir, conf.frame_extension)

    gt = utls.getPositives(gtFileNames)

    for i, s in enumerate(sp_scales):
        print('Scale {}, {}/{}'.format(s, i+1, len(sp_scales)))
        my_dataset.conf.slic_size = s

        labels = spix_extr.extract(conf.frameFileNames,
                            os.path.join(conf.precomp_desc_path,
                                        'sp_labels.npz'),
                            conf.slic_compactness,
                            conf.slic_size,
                            False)

        import pdb; pdb.set_trace()
        gt_sp = utls.make_sp_gts(gt, labels, pos_thr=0)
        f1 = f1_score(gt.ravel(), gt_sp.ravel())

        fileOut = os.path.join(conf.dataOutDir,
                            'results_scale_{}.npz'.format(s))
        data = dict()
        data['labels'] = labels
        data['f1'] = f1
        data['gt_sp'] = gt_sp
        data['scale'] = s
        np.savez(fileOut, **data)
