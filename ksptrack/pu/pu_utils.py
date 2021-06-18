#!/usr/bin/env python3

from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from ksptrack.pu.im_utils import get_features
from skimage.filters import threshold_multiotsu


def trans_fn(state, noise, xi, filt_size, dt):
    filt_size = int(2 * np.round(filt_size * state.size / 2)) + 1
    state = smooth(state, window_len=filt_size)
    new_state = state - xi * dt
    new_state += noise

    return new_state


def obs_fn(state, noise, xi):
    return state + noise - xi


def x_mean_fn(sigmas, Wm, ub, lb):
    x = np.dot(Wm, sigmas)
    x = np.clip(x, a_min=lb, a_max=ub)
    return x


def get_curr_xi(pi0, xi_fac_start, xi_fac_end, t, T):
    xi_start = pi0 * xi_fac_start
    xi_end = pi0 * xi_fac_end
    return xi_start + (xi_end - xi_start) * t / T


def init_kfs(n_frames, init_mean, init_cov, cfg):

    trans_cov = np.eye(n_frames) * cfg.trans_fac
    obs_cov = np.eye(n_frames) * cfg.obs_fac

    init_means = np.ones(n_frames) * init_mean
    init_covs = np.eye(n_frames) * init_cov

    xi = get_curr_xi(cfg.init_pi, cfg.xi_fac_start, cfg.xi_fac_end, 0,
                     cfg.epochs_pred)

    trans_fn_ = lambda s, dt: trans_fn(s, 0, xi, cfg.pi_filt_size, dt)
    obs_fn_ = lambda s: obs_fn(s, 0, 0)
    x_mean_fn_ = lambda sigmas, Wm: x_mean_fn(sigmas, Wm, cfg.init_pi, 0)

    points = MerweScaledSigmaPoints(n=n_frames, alpha=1e-3, beta=2, kappa=0)

    filter_ = UKF(n_frames,
                  n_frames,
                  1,
                  obs_fn_,
                  trans_fn_,
                  points=points,
                  x_mean_fn=x_mean_fn_)
    filter_.Q *= cfg.trans_fac
    filter_.R *= cfg.obs_fac
    filter_.x = init_mean * np.ones(n_frames)
    filter_.P *= init_cov
    filter_.predict()

    return filter_, [init_means for f in range(n_frames)
                     ], [init_covs for f in range(n_frames)]


def get_model_freqs(probas, n_classes=3):
    freqs = []

    pbar = tqdm.tqdm(total=len(probas))
    for prob in probas:
        thr = threshold_multiotsu(prob, classes=n_classes)[0]

        freqs.append((prob >= thr).sum() / prob.size)

        pbar.set_description('[obsrv]')
        pbar.update(1)

    pbar.close()

    return freqs


def update_priors_kf(model, dl, device, state_means, state_covs, filter,
                     writer, out_path_plots, out_path_data, epoch, cfg):

    res = get_features(model, dl, device)
    probas = res['outs_unpooled'] if cfg.pxls else res['outs']
    truths = res['truths_unpooled'] if cfg.pxls else res['truths']
    truths = res['truths_unpooled'] if cfg.pxls else res['truths']

    clf = np.array([(p >= 0.5).sum() / p.size for p in probas])

    observations = np.array([(p**cfg.gamma).mean() for p in probas])
    observations = np.clip(observations, a_min=0., a_max=cfg.init_pi)

    filter.update(observations)
    filter.predict()

    # update "velocity"
    new_xi = get_curr_xi(cfg.init_pi, cfg.xi_fac_start, cfg.xi_fac_end, epoch,
                         cfg.epochs_pred)

    print('new_xi: ', new_xi)
    trans_fn_ = lambda s, dt: trans_fn(s, 0, new_xi, cfg.pi_filt_size, dt)
    filter.fx = trans_fn_

    state_means.append(filter.x)
    state_covs.append(filter.P)

    true_pi = np.array([np.sum(p >= 0.5) / p.size for p in truths])

    print('new state_mean max: ', state_means[-1].max())

    sns.set_style('darkgrid')
    df = pd.DataFrame.from_dict({
        'frame': np.arange(true_pi.size),
        'true': true_pi,
        'new_xi': new_xi,
        'observ.': observations,
        'clf': clf,
        'priors_max': state_means[0],
        'priors_t': state_means[-2],
        'priors_t+1': state_means[-1]
    })
    df.to_pickle(pjoin(out_path_data, 'ep_{:04d}.p'.format(epoch)))
    df = df.drop(columns=['new_xi'])
    g = sns.lineplot(x='frame',
                     y='value',
                     hue='variable',
                     data=pd.melt(df, ['frame']))
    plt.ylim(0, 1.1 * cfg.init_pi)
    plt.savefig(pjoin(out_path_plots, 'ep_{:04d}.png'.format(epoch)))
    plt.close()

    # variance of predictions on pseudo-negatives
    vars_pseudo_negs = np.mean([np.var(p[p < 0.5]) for p in probas])

    err = np.mean(np.abs(true_pi - state_means[-1]))

    writer.add_scalar('prior_err/{}'.format(cfg.exp_name), err, epoch)
    writer.add_scalar('var_pseudo_neg/{}'.format(cfg.exp_name),
                      vars_pseudo_negs, epoch)

    return filter, state_means, state_covs


class MyMerweScaledSigmaPoints(MerweScaledSigmaPoints):
    def __init__(self, upperbound, lowerbound, *args, **kwargs):
        super(MyMerweScaledSigmaPoints, self).__init__(*args, **kwargs)
        self.upperbound = upperbound
        self.lowerbound = lowerbound

    def sigma_points(self, x, P):

        sigma_points = super(MyMerweScaledSigmaPoints, self).sigma_points(x, P)
        sigma_points = np.clip(sigma_points,
                               a_min=self.lowerbound,
                               a_max=self.upperbound)

        return sigma_points


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.pad(x, window_len // 2, mode='edge')
    #print(len(s))
    if window == 'flat':  #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y
