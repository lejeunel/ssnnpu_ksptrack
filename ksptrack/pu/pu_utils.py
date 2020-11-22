#!/usr/bin/env python3

import numpy as np
from ksptrack.pu.im_utils import get_features
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt
from pykalman import KalmanFilter, AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter
from pykalman.unscented import augmented_unscented_filter_points
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from scipy.signal import medfilt


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
                     cfg.epochs_pred - cfg.epochs_pre_pred)

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


def update_priors_kf(model, dl, device, state_means, state_covs, filter,
                     writer, out_path_plots, out_path_data, epoch, cfg):

    res = get_features(model, dl, device, loc_prior=cfg.loc_prior)
    probas = res['outs_unpooled'] if cfg.pxls else res['outs']
    truths = res['truths_unpooled'] if cfg.pxls else res['truths']
    probas = [p - p.min() for p in probas]
    probas = [p / p.max() for p in probas]
    truths = res['truths_unpooled'] if cfg.pxls else res['truths']

    freq = np.array([np.sum(p >= 0.5) / p.size for p in probas])

    filter.update(np.clip(freq, a_min=0., a_max=cfg.init_pi))
    filter.predict()

    # update "velocity"
    new_xi = get_curr_xi(cfg.init_pi, cfg.xi_fac_start, cfg.xi_fac_end,
                         epoch - cfg.epochs_pre_pred,
                         cfg.epochs_pred - cfg.epochs_pre_pred)

    print('new_xi: ', new_xi)
    trans_fn_ = lambda s, dt: trans_fn(s, 0, new_xi, cfg.pi_filt_size, dt)
    filter.fx = trans_fn_

    state_means.append(filter.x)
    state_covs.append(filter.P)

    true_pi = np.array([np.sum(p >= 0.5) / p.size for p in truths])

    print('new state_mean max: ', state_means[-1].max())
    print('true max: ', true_pi.max())

    sns.set_style('darkgrid')
    df = pd.DataFrame.from_dict({
        'frame': np.arange(true_pi.size),
        'true': true_pi,
        'new_xi': new_xi,
        'model': freq,
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


def update_priors(model,
                  dl,
                  device,
                  training_priors,
                  writer,
                  out_path_plots,
                  out_path_data,
                  epoch,
                  cfg,
                  grad_method='clip',
                  inval_mode='copy',
                  scale_grad=0.05,
                  decrease_pimax=True):

    res = get_features(model, dl, device, loc_prior=cfg.loc_prior)
    probas = res['outs_unpooled'] if cfg.pxls else res['outs']
    probas = [p - p.min() for p in probas]
    probas = [p / p.max() for p in probas]
    truths = res['truths_unpooled'] if cfg.pxls else res['truths']

    new_priors = np.array(
        [np.sum(cfg.pi_mul * p >= 0.5) / p.size for p in probas])

    d_priors = cfg.pi_eta * (new_priors - training_priors[-1])

    if grad_method == 'clip':
        d_priors = np.clip(d_priors, a_min=-cfg.pi_min, a_max=cfg.pi_min)
    else:
        d_priors = np.sign(d_priors) * np.abs(d_priors) * scale_grad

    idx_invalids = new_priors > training_priors[0]
    idx_valids = np.logical_not(idx_invalids)
    new_priors[
        idx_valids] = training_priors[-1][idx_valids] + d_priors[idx_valids]

    if inval_mode == 'copy':
        new_priors[
            idx_invalids] = training_priors[-1][idx_invalids] - cfg.pi_xi
    elif idx_invalids.sum() > 0:
        training_priors[0][idx_invalids] = np.max(np.vstack(
            (training_priors[0][idx_invalids] - cfg.pi_xi,
             cfg.init_pi * np.ones(idx_invalids.sum()) * 0.5)),
                                                  axis=0)
        new_priors[idx_invalids] = training_priors[0][idx_invalids]

    if decrease_pimax:
        training_priors[0][idx_invalids] = np.clip(
            training_priors[0][idx_invalids] - cfg.pi_xi,
            a_min=cfg.pi_min,
            a_max=None)

    # frames that exceed pi_max are re-initialized or kept as before
    # new_priors[idx_invalids] = training_priors[-1][idx_invalids] - cfg.pi_xi

    freq = np.array([np.sum(p >= 0.5) / p.size for p in probas])
    training_priors.append(new_priors)

    true_pi = np.array([np.sum(p >= 0.5) / p.size for p in truths])

    sns.set_style('darkgrid')
    df = pd.DataFrame.from_dict({
        'frame': np.arange(true_pi.size),
        'true': true_pi,
        'model': freq,
        'priors_max': training_priors[0],
        'priors_i-1': training_priors[-2],
        'priors_i': new_priors
    })
    g = sns.lineplot(x='frame',
                     y='value',
                     hue='variable',
                     data=pd.melt(df, ['frame']))
    plt.ylim(0, 1.1 * cfg.init_pi)
    plt.savefig(pjoin(out_path_plots, 'ep_{:04d}.png'.format(epoch)))
    plt.close()

    # variance of predictions on pseudo-negatives
    vars_pseudo_negs = np.mean([np.var(p[p < 0.5]) for p in probas])

    df['d_priors'] = d_priors
    df.to_pickle(pjoin(out_path_data, 'ep_{:04d}.p'.format(epoch)))

    err = np.mean(np.abs(true_pi - new_priors))

    writer.add_scalar('prior_err/{}'.format(cfg.exp_name), err, epoch)
    writer.add_scalar('var_pseudo_neg/{}'.format(cfg.exp_name),
                      vars_pseudo_negs, epoch)

    return training_priors


from pykalman.standard import _last_dims, _determine_dimensionality, _arg_or_default
from pykalman.unscented import Moments, moments2points, unscented_filter_correct, unscented_filter_predict
from pykalman.utils import array1d, array2d, check_random_state, get_params, preprocess_arguments, check_random_state


class ConstrainedUKF(UnscentedKalmanFilter):
    def __init__(self, upperbound, lowerbound, *args, **kwargs):
        super(ConstrainedUKF, self).__init__(*args, **kwargs)
        self.upperbound = upperbound
        self.lowerbound = lowerbound

    def clip(self, sp):
        clipped = np.clip(sp.points,
                          a_min=self.lowerbound,
                          a_max=self.upperbound)

        return sp._replace(points=clipped)

    def filter_update(self,
                      filtered_state_mean,
                      filtered_state_covariance,
                      observation=None,
                      transition_function=None,
                      transition_covariance=None,
                      observation_function=None,
                      observation_covariance=None):
        r"""Update a Kalman Filter state estimate

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Parameters
        ----------
        filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t given observations from times
            [1...t]
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t given observations from
            times [1...t]
        observation : [n_dim_obs] array or None
            observation from time t+1.  If `observation` is a masked array and
            any of `observation`'s components are masked or if `observation` is
            None, then `observation` will be treated as a missing observation.
        transition_function : optional, function
            state transition function from time t to t+1.  If unspecified,
            `self.transition_functions` will be used.
        transition_covariance : optional, [n_dim_state, n_dim_state] array
            state transition covariance from time t to t+1.  If unspecified,
            `self.transition_covariance` will be used.
        observation_function : optional, function
            observation function at time t+1.  If unspecified,
            `self.observation_functions` will be used.
        observation_covariance : optional, [n_dim_obs, n_dim_obs] array
            observation covariance at time t+1.  If unspecified,
            `self.observation_covariance` will be used.

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t+1 given observations from times
            [1...t+1]
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t+1 given observations
            from times [1...t+1]
        """
        # initialize parameters
        (transition_functions, observation_functions, transition_cov,
         observation_cov, _, _) = (self.__initialize_parameters())

        def default_function(f, arr):
            if f is None:
                assert len(arr) == 1
                f = arr[0]
            return f

        transition_function = default_function(transition_function,
                                               transition_functions)
        observation_function = default_function(observation_function,
                                                observation_functions)
        transition_covariance = _arg_or_default(transition_covariance,
                                                transition_cov, 2,
                                                "transition_covariance")
        observation_covariance = _arg_or_default(observation_covariance,
                                                 observation_cov, 2,
                                                 "observation_covariance")

        # Make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        # make sigma points
        (points_state, points_transition,
         points_observation) = (augmented_unscented_filter_points(
             filtered_state_mean, filtered_state_covariance,
             transition_covariance, observation_covariance))

        # project both points and moments
        points_state = self.clip(points_state)

        # predict
        (points_pred,
         moments_pred) = (unscented_filter_predict(transition_function,
                                                   points_state,
                                                   points_transition))

        # project again
        points_pred = self.clip(points_pred)
        # moments_pred = self.clip(moments_pred)

        # correct
        next_filtered_state_mean, next_filtered_state_covariance = (
            unscented_filter_correct(observation_function,
                                     moments_pred,
                                     points_pred,
                                     observation,
                                     points_observation=points_observation))

        # ... and project again
        next_filtered_state_mean = np.clip(next_filtered_state_mean,
                                           a_min=self.lowerbound,
                                           a_max=self.upperbound)

        return (next_filtered_state_mean, next_filtered_state_covariance)

    def __initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults"""

        arguments = get_params(super(ConstrainedUKF, self))
        defaults = self._default_parameters()
        converters = self._converters()

        processed = preprocess_arguments([arguments, defaults], converters)
        return (processed['transition_functions'],
                processed['observation_functions'],
                processed['transition_covariance'],
                processed['observation_covariance'],
                processed['initial_state_mean'],
                processed['initial_state_covariance'])


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
