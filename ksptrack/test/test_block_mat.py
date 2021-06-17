from ksptrack.hoof_extractor import HOOFExtractor
from ksptrack.utils.data_manager import DataManager
from ksptrack.sp_manager import SuperpixelManager
from ksptrack.cfgs import cfg
import matplotlib.pyplot as plt
import numpy as np

#ratio = 0.1 # gives 100 sps
ratio = 0.07 # gives 220 sps

cfg_path = '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling/DatasetTest/results/2018-08-24_13-18-32_exp/cfg.yml'

conf = cfg.load_and_convert(cfg_path)
dm = DataManager(conf)

flows = dm.get_flows()
labels = dm.get_labels()
shape = labels[..., 0].shape

sp_man = SuperpixelManager(dm, conf, with_flow=True)

