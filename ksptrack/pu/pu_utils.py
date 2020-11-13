#!/usr/bin/env python3

import numpy as np
from ksptrack.pu.im_utils import get_features
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


def init_kfs(n_frames,
             n_bins,
             init_mean,
             init_cov,
             trans_factor,
             trans_offset=0.,
             trans_cov=0.01,
             obs_factor=1.,
             obs_offset=0.,
             obs_cov=0.01):

    trans_matrix = np.eye(n_bins) * trans_factor
    trans_off = np.ones(n_bins) * trans_offset
    trans_cov = np.ones(n_bins) * trans_cov

    obs_matrix = np.eye(n_bins) * obs_factor
    obs_off = np.ones(n_bins) * obs_offset
    obs_cov = np.ones(n_bins) * obs_cov

    init_means = np.ones(n_bins) * init_mean
    init_covs = np.ones(n_bins) * init_cov

    filters = [
        KalmanFilter(trans_matrix, obs_matrix, trans_cov, obs_cov, trans_off,
                     obs_off, init_means, init_covs) for _ in range(n_frames)
    ]

    return filters, [init_means for f in range(n_frames)
                     ], [init_covs for f in range(n_frames)]


def update_priors_kf(model,
                     dl,
                     device,
                     n_bins,
                     training_priors,
                     state_means,
                     state_covs,
                     filters,
                     model_priors,
                     writer,
                     out_path_plots,
                     out_path_data,
                     epoch,
                     cfg,
                     decrease_pimax=True):

    res = get_features(model, dl, device, loc_prior=cfg.loc_prior)
    probas = res['outs_unpooled'] if cfg.pxls else res['outs']
    truths = res['truths_unpooled'] if cfg.pxls else res['truths']

    new_priors = []
    for state_mean, state_cov, filt, pi_hat in zip(state_means, state_covs,
                                                   filters, probas):
        dist_pihat = np.histogram(pi_hat, n_bins=n_bins)
        state_mean, state_cov = filt.filter_update(state_mean, state_cov,
                                                   pi_hat)
        new_priors.append(np.linspace(0, 1, n_bins)[state_mean.argmax()])

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

    return training_priors, model_priors


def update_priors(model,
                  dl,
                  device,
                  training_priors,
                  model_priors,
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
        new_priors[idx_invalids] = training_priors[-1][idx_invalids]
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

    return training_priors, model_priors
