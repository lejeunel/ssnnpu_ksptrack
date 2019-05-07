import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import logging
from ksptrack.utils import my_utils as utls
import torch
from pytorch_utils.loader import Loader
from dqnksp.episode import Episode
from dqnksp import utils as dqn_utls
from dqnksp.dqnshape import DQNShape
from imgaug import augmenters as iaa
from itertools import count


class EntranceAgent:
    def __init__(self, cfg, cfg_agent, actions, checkpoint_path):

        self.logger = logging.getLogger('EntranceAgent')

        self.im_paths = cfg.frameFileNames
        self.cfg_agent = cfg_agent
        self.cfg = cfg

        # initial region parameters
        self.init_circle_rel_radius = self.cfg_agent.init_circle_rel_radius
        self.expand_n_pix = self.cfg_agent.expand_n_pix

        self.in_shape = self.cfg_agent.in_shape
        self.model_path = checkpoint_path
        self.actions = actions
        self.n_outputs = len(actions)

        # Make image transforms for network
        transf_shape = {
            'image': [self.cfg_agent.in_shape] * 2,
            'truth': [self.cfg_agent.in_shape] * 2
        }
        transf_normalize = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        self.state_trans_fun = lambda s: dqn_utls.transform_state(
            s,
            transf_shape,
            transf_normalize,
            one_hot_encode=True,
            hist_size=self.cfg_agent.history_size)

        # Make image loader
        self.loader = Loader(cfg,
                             cfg.root_path,
                             cfg_agent.truth_type)

        self.im_shape = self.loader[0][0].shape

        # Load model
        self.logger.info('Loading checkpoint: {}'.format(checkpoint_path))
        cp = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        self.model = DQNShape(
            self.n_outputs,
            history_size=self.cfg_agent.history_size,
            pretrained_weights_path=None,
            cuda=False)
        self.model.load_state_dict(cp['state_dict'])

        self.device = torch.device('cpu')
        self.model.eval()

    def get_actions(self, x, y, f_num, init_actions=[], max_actions=50):

        im, _ = self.loader[f_num]

        loc = utls.norm_to_pix(x, y, im.shape[1], im.shape[0])
        loc_flat = np.ravel_multi_index(loc, im.shape[0:2])

        self.ep = Episode(self.cfg_agent,
                          im,
                          loc_flat,
                          init_history=init_actions,
                          expand_n_pix=self.expand_n_pix)

        state = self.ep.get_state()
        feat = self.model.get_feats([state.apply_fun(self.state_trans_fun)])
        state.feat = feat.cpu().detach().numpy()

        for i in count():
            # plt.imshow(state.masks[..., 0, 0]);plt.show()

            with torch.no_grad():
                # action = self.model.get_actions([
                #     state.apply_fun(self.state_trans_fun)
                # ]).argmax(1)[0].cpu().detach().numpy()
                q_values = self.model.get_actions([
                    state.apply_fun(self.state_trans_fun)
                ]).cpu().detach().numpy()
                print(q_values)
                action = np.argmax(q_values)

            next_state = self.ep.get_state()

            feat = self.model.get_feats(
                [next_state.apply_fun(self.state_trans_fun)])
            next_state.feat = feat.cpu().detach().numpy()


            done = self.ep.step(dqn_utls.action_int_to_str(action))

            state = next_state

            if (done):
                # self.ep.history = self.ep.history[0:-1]
                break
            if (dqn_utls.detect_oscillation(self.ep.history)):
                self.ep.history = self.ep.history[0:-1]
                break
            if (i >= max_actions):
                break

            # print(ep.history)

        return self.ep.history, self.ep.get_state()

    def make_mask(self, actions):
        init_circle_rel_radius = self.cfg_agent.init_circle_rel_radius
        expand_rel_magn = self.cfg_agent.expand_rel_magn
        return dqn_utls.actions_to_region(actions, init_circle_rel_radius,
                                          expand_rel_magn, max(self.im_shape))


def make_patch(im, loc, patch_size):

    patch_size += not patch_size % 2

    loc_flat = np.ravel_multi_index(loc, im.shape[0:2])
    patch = dqn_utls.extract_patch(im, patch_size, loc_flat)[..., np.newaxis]

    return patch
