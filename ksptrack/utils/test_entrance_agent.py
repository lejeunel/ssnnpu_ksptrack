from ksptrack.utils.entrance_agent import EntranceAgent
from ksptrack.exps import results_dirs as rd
from ksptrack.cfgs.cfg import load_and_convert
from os.path import join as pjoin
import os
from dqnksp import utils as utls
import matplotlib.pyplot as plt
import pandas as pd
from progressbar import ProgressBar as pbar
import numpy as np
from skimage import (io, draw)

type_ = 'Tweezer'
# type_ = 'Slitlamp'
# type_ = 'Brain'

seq_num = 0
seq = 0
frame_suffix = 'squeezenet_bounded'
# frame_suffix = 'sym_reward_beta_0_1'
max_actions = 15

run_path = 'dqn/tweezer/runs/2019-03-25_16-34-13'

res_path = pjoin(rd.root_dir, rd.res_dirs_dict_ksp[type_][seq_num][seq])
cfg = load_and_convert(pjoin(res_path, 'cfg.yml'))

agent_model_path = pjoin(rd.root_dir,
                         run_path,
                         'checkpoint.pth.tar')
# agent_model_path = pjoin(rd.root_dir, 'dqn/pascal/model_beta_0_1.pth.tar')

cfg_agent_path = pjoin(rd.root_dir,
                       run_path,
                       'cfg.yml')

cfg_agent = load_and_convert(cfg_agent_path)
agent = EntranceAgent(cfg,
                      cfg_agent,
                      utls.Actions,
                      agent_model_path)

locs2d = pd.read_csv(
    pjoin(cfg.root_path, cfg.ds_dir, cfg.locs_dir, cfg.csvFileName_fg),
    skiprows=4,
    delimiter=';')

out_dir = pjoin(rd.root_dir,
                'dqn/test/{}_{}_{}'.format(type_,
                                           frame_suffix,
                                           seq_num))

if(not os.path.exists(out_dir)):
    os.makedirs(out_dir)

actions_seq = []
states = []
imgs = []

for index, row in locs2d.iterrows():

    f = int(row['frame'])
    if (index == 0):
        # set empty initial action vector
        init_actions = []
    else:
        init_actions = actions_seq[-1]

    actions, state = agent.get_actions(
        row['x'], row['y'], f, init_actions=[],
        max_actions=max_actions)
    actions_seq.append(actions)
    states.append(state)

    print('({}/{}), n_actions: {}, actions: {}'.format(index+1,
                                                       locs2d.shape[0],
                                                       len(actions),
                                                       actions))

    im = agent.ep.img
    im = agent.ep.bbox.apply_mask(im/255)
    rr, cc = draw.circle(*agent.ep.loc, 8)
    im[rr, cc, :] = (0, 1, 0)

    io.imsave(pjoin(out_dir, 'frame_{}.png'.format(index)), im)
    imgs.append(im)
