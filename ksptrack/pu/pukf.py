#!/usr/bin/env python3
from pykalman import UnscentedKalmanFilter


class PUKalmanFilter:
    """
    """
    def __init__(self, n_frames, init_state_mean, init_state_cov,
                 transition_cov, observation_cov, xi):
        """
        """

        self.n_frames = n_frames
        self.init_state_mean = init_state_mean
        self.init_state_cov = init_state_cov
        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.observations = []
        self.xi = xi
        self.pi_max = init_state_mean

    def trans_fn(self, state, noise):
        idx_invalids = state > self.pi_max
        self.pi_max

    def add_obs(self, obs):
        self.observations.append(obs)
