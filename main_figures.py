#!/usr/bin/env python
# coding: utf-8
import argparse
from custom_cmap import custom_cmap
from copy import deepcopy as copy
import einops
import fit_trunc_dist
from functools import partial
import glob
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import numpy as np
from numpy import sqrt, pi, exp
from numba import jit, prange
from parse_swag_args import parse
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import truncnorm
from scipy.special import erf
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
import spock_reg_model
import sys
import torch
import time
from tqdm.notebook import tqdm
import utils
import wandb
import petit
import pickle
import modules

# to fix the fonts?
plt.rcParams.update(plt.rcParamsDefault)



def calc_scores(args, checkpoint_filename, logger=None, plot_random=False):
    s = checkpoint_filename + "*output.pkl"
    swag_ensemble = [
        spock_reg_model.load_swag(fname).cuda()
        for fname in glob.glob(s) #
    ]

    if len(swag_ensemble) == 0:
        raise ValueError(s + " not found!")


    if plot_random:
        checkpoint_filename += '_random'


    plt.switch_backend('agg')


    colorstr = """*** Primary color:

       shade 0 = #A0457E = rgb(160, 69,126) = rgba(160, 69,126,1) = rgb0(0.627,0.271,0.494)
       shade 1 = #CD9CBB = rgb(205,156,187) = rgba(205,156,187,1) = rgb0(0.804,0.612,0.733)
       shade 2 = #BC74A1 = rgb(188,116,161) = rgba(188,116,161,1) = rgb0(0.737,0.455,0.631)
       shade 3 = #892665 = rgb(137, 38,101) = rgba(137, 38,101,1) = rgb0(0.537,0.149,0.396)
       shade 4 = #74104F = rgb(116, 16, 79) = rgba(116, 16, 79,1) = rgb0(0.455,0.063,0.31)

    *** Secondary color (1):

       shade 0 = #CDA459 = rgb(205,164, 89) = rgba(205,164, 89,1) = rgb0(0.804,0.643,0.349)
       shade 1 = #FFE9C2 = rgb(255,233,194) = rgba(255,233,194,1) = rgb0(1,0.914,0.761)
       shade 2 = #F1D195 = rgb(241,209,149) = rgba(241,209,149,1) = rgb0(0.945,0.82,0.584)
       shade 3 = #B08431 = rgb(176,132, 49) = rgba(176,132, 49,1) = rgb0(0.69,0.518,0.192)
       shade 4 = #956814 = rgb(149,104, 20) = rgba(149,104, 20,1) = rgb0(0.584,0.408,0.078)

    *** Secondary color (2):

       shade 0 = #425B89 = rgb( 66, 91,137) = rgba( 66, 91,137,1) = rgb0(0.259,0.357,0.537)
       shade 1 = #8C9AB3 = rgb(140,154,179) = rgba(140,154,179,1) = rgb0(0.549,0.604,0.702)
       shade 2 = #697DA0 = rgb(105,125,160) = rgba(105,125,160,1) = rgb0(0.412,0.49,0.627)
       shade 3 = #294475 = rgb( 41, 68,117) = rgba( 41, 68,117,1) = rgb0(0.161,0.267,0.459)
       shade 4 = #163163 = rgb( 22, 49, 99) = rgba( 22, 49, 99,1) = rgb0(0.086,0.192,0.388)

    *** Complement color:

       shade 0 = #A0C153 = rgb(160,193, 83) = rgba(160,193, 83,1) = rgb0(0.627,0.757,0.325)
       shade 1 = #E0F2B7 = rgb(224,242,183) = rgba(224,242,183,1) = rgb0(0.878,0.949,0.718)
       shade 2 = #C9E38C = rgb(201,227,140) = rgba(201,227,140,1) = rgb0(0.788,0.89,0.549)
       shade 3 = #82A62E = rgb(130,166, 46) = rgba(130,166, 46,1) = rgb0(0.51,0.651,0.18)
       shade 4 = #688C13 = rgb(104,140, 19) = rgba(104,140, 19,1) = rgb0(0.408,0.549,0.075)"""

    colors = []
    shade = 0
    for l in colorstr.replace(' ', '').split('\n'):
        elem = l.split('=')
        if len(elem) != 5: continue
        if shade == 0:
            new_color = []
        rgb = lambda x, y, z: np.array([x, y, z]).astype(np.float32)

        new_color.append(eval(elem[2]))

        shade += 1
        if shade == 5:
            colors.append(np.array(new_color))
            shade = 0
    colors = np.array(colors)/255.0

    swag_ensemble[0].make_dataloaders()
    if plot_random:
        assert swag_ensemble[0].ssX is not None
        tmp_ssX = copy(swag_ensemble[0].ssX)
        # print(tmp_ssX.mean_)
        # if args.train_all:
        #     swag_ensemble[0].make_dataloaders(
        #         ssX=swag_ensemble[0].ssX,
        #         train=True,
        #         plot_random=True)
        # else:
        swag_ensemble[0].make_dataloaders(
            ssX=swag_ensemble[0].ssX,
            train=False,
            plot_random=True) #train=False means we show the whole dataset (assuming we don't train on it!)

        # print(swag_ensemble[0].ssX.mean_)
        assert np.all(tmp_ssX.mean_ == swag_ensemble[0].ssX.mean_)

    val_dataloader = swag_ensemble[0]._val_dataloader

    def sample_full_swag(X_sample):
        """Pick a random model from the ensemble and sample from it
        within each model, it samples from its weights."""

        swag_i = np.random.randint(0, len(swag_ensemble))
        swag_model = swag_ensemble[swag_i]
        swag_model.eval()
        swag_model.w_avg = swag_model.w_avg.cuda()
        swag_model.w2_avg = swag_model.w2_avg.cuda()
        swag_model.pre_D = swag_model.pre_D.cuda()
        swag_model.cuda()
        out = swag_model.forward_swag(X_sample, scale=0.5)
        return out

    truths = []
    preds = []
    raw_preds = []

    nc = 0
    losses = 0.0
    do_sample = True
    for X_sample, y_sample in tqdm(val_dataloader):
        X_sample = X_sample.cuda()
        y_sample = y_sample.cuda()
        nc += len(y_sample)
        truths.append(y_sample.cpu().detach().numpy())

        raw_preds.append(
            np.array([sample_full_swag(X_sample).cpu().detach().numpy() for _ in range(2000)])
        )

    truths = np.concatenate(truths)

    _preds = np.concatenate(raw_preds, axis=1)

    # numpy sampling is way too slow:


    def fast_truncnorm(
            loc, scale, left=np.inf, right=np.inf,
            d=10000, nsamp=50, seed=0):
        """Fast truncnorm sampling.

        Assumes scale and loc have the desired shape of output.
        length is number of elements.
        Select nsamp based on expecting at last one sample
            to fit within your (left, right) range.
        Select d based on memory considerations - need to operate on
            a (d, nsamp) array.
        """
        oldscale = scale
        oldloc = loc

        scale = scale.reshape(-1)
        loc = loc.reshape(-1)
        samples = np.zeros_like(scale)
        start = 0

        for start in range(0, scale.shape[0], d):

            end = start + d
            if end > scale.shape[0]:
                end = scale.shape[0]

            cd = end-start
            rand_out = np.random.randn(
                nsamp, cd
            )

            rand_out = (
                rand_out * scale[None, start:end]
                + loc[None, start:end]
            )

            #rand_out is (nsamp, cd)
            if right == np.inf:
                mask = (rand_out > left)
            elif left == np.inf:
                mask = (rand_out < right)
            else:
                mask = (rand_out > left) & (rand_out < right)

            first_good_val = rand_out[
                mask.argmax(0), np.arange(cd)
            ]

            samples[start:end] = first_good_val

        return samples.reshape(*oldscale.shape)

    std = _preds[..., 1]
    mean = _preds[..., 0]

    loc = mean
    scale = std

    sample_preds = np.array(
            fast_truncnorm(np.array(mean), np.array(std),
                   left=4, d=874000, nsamp=40));

    stable_past_9 = sample_preds >= 9


    _prior = lambda logT: (
        3.27086190404742*np.exp(-0.424033970670719 * logT) -
        10.8793430454878*np.exp(-0.200351029031774 * logT**2)
    )
    normalization = quad(_prior, a=9, b=np.inf)[0]

    prior = lambda logT: _prior(logT)/normalization

    # Let's generate random samples of that prior:
    n_samples = stable_past_9.sum()
    bins = n_samples*4
    top = 100.
    bin_edges = np.linspace(9, top, num=bins)
    cum_values = [0] + list(np.cumsum(prior(bin_edges)*(bin_edges[1] - bin_edges[0]))) + [1]
    bin_edges = [9.] +list(bin_edges)+[top]
    inv_cdf = interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    samples = inv_cdf(r)

    sample_preds[stable_past_9] = samples

    _preds.shape

    # # expectation of samples
    # preds = np.average(sample_preds, 0)
    # stds = np.std(sample_preds, 0)

    # # median of samples
    # preds = np.median(sample_preds, 0)
    # stds = (
    #     (lambda x: 0.5*(np.percentile(x, q=50 + 68/2, axis=0) - np.percentile(x, q=50-68/2, axis=0)))
    #     (sample_preds)
    # )

    # # median of dists
    preds = np.median(_preds[..., 0], 0)
    stds = np.median(_preds[..., 1], 0)

    # # fit a truncated dist using avg, var
    # tmp = fit_trunc_dist.find_mu_sig(sample_preds.T)
    # preds = tmp[:, 0]
    # stds = tmp[:, 1]

    # # with likelihood (slow)
    # tmp = fit_trunc_dist.find_mu_sig_likelihood(sample_preds[:300, :].T)
    # preds = tmp[:, 0]
    # stds = tmp[:, 1]

    # weighted average of mu
    # w_i = 1/_preds[:, :, 1]**2
    # w_i /= np.sum(w_i, 0)
    # preds = np.average(_preds[:, :, 0], 0, weights=w_i)
    # stds = np.average(_preds[:, :, 1]**2, 0)**0.5

    # Check that confidence intervals are satisifed. Calculate mean and std of samples. Take abs(truths - mean)/std = sigma. The CDF of this distrubtion should match that of a Gaussian. Otherwise, rescale "scale".

    tmp_mask = (truths > 6) & (truths < 7) #Take this portion since its far away from truncated parts
    averages = preds#np.average(sample_preds, 0)
    gaussian_stds = stds#np.std(sample_preds, 0)
    sigma = (truths[tmp_mask] - np.tile(averages, (2, 1)).T[tmp_mask])/np.tile(gaussian_stds, (2, 1)).T[tmp_mask]

    np.save(checkpoint_filename + 'model_error_distribution.npy', sigma)

    bins = 30
    fig = plt.figure(figsize=(4, 4))
    plt.hist(np.abs(sigma), bins=bins, range=[0, 2.5], density=True,
                color=colors[0, 3],
             alpha=1, label='Model error distribution')
    np.random.seed(0)
    plt.hist(np.abs(np.random.randn(len(sigma))), bins=bins, range=[0, 2.5], density=True,
                color=colors[1, 3],
             alpha=0.5, label='Gaussian distribution')
    plt.ylim(0, 1.2)
    plt.ylabel('Density', fontsize=14)
    plt.xlabel('Error over sigma', fontsize=14)
    # plt.xlabel('$|\mu_θ - y|/\sigma_θ$', fontsize=14)
    plt.legend()
    fig.savefig(checkpoint_filename + 'error_dist.pdf')


    # Looks great! We didn't even need to tune it. Just use the same scale as the paper (0.5). Perhaps, however, with epistemic uncertainty, we will need to tune.


    def density_scatter(x, y, xlabel='', ylabel='', clabel='Sample Density', log=False,
        width_mult=1, bins=30, p_cut=None, update_rc=True, ax=None, fig=None, cmap='viridis', **kwargs):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        xy = np.array([x, y]).T
        px = xy[:, 0]
        py = xy[:, 1]
        if p_cut is not None:
            p = p_cut
            range_x = [np.percentile(xy[:, 0], i) for i in [p, 100-p]]
            range_y = [np.percentile(xy[:, 1], i) for i in [p, 100-p]]
            pxy = xy[(xy[:, 0] > range_x[0]) & (xy[:, 0] < range_x[1]) & (xy[:, 1] > range_y[0]) & (xy[:, 1] < range_y[1])]
        else:
            pxy = xy
        px = pxy[:, 0]
        py = pxy[:, 1]
        norm = None
        if log:
            norm = LogNorm()

        h, xedge, yedge, im = ax.hist2d(
            px, py, density=True, norm=norm,
            bins=[int(width_mult*bins), bins], cmap=cmap, **kwargs)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.colorbar(im, ax=ax).set_label(clabel)
        fig.tight_layout()

        return fig, ax

    _preds.shape

    #confidences_to_plot = 'low med high vhigh vvhigh'.split(' ')
    confidences_to_plot = ['low']

    # %matplotlib inline


    show_transparency = True

    main_shade = 3
    main_color = colors[2, main_shade]
    off_color = colors[2, main_shade]


    plt.style.use('default')
    sns.set_style('white')
    plt.rc('font', family='serif')

    # +
    for confidence in confidences_to_plot:
        py = preds
        py = np.clip(py, 4, 9)
        px = np.average(truths, 1)

        mask = np.all(truths < 9.99, 1) # np.all(truths < 8.99, 1)

        if confidence != 'low':
            #tmp_std = np.std(sample_preds, 0)/py
            tmp_std = stds/py
            if confidence == 'high':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 50))
            elif confidence == 'vhigh':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 25))
            elif confidence == 'vvhigh':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 10))
            elif confidence == 'med':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 70))

        ppx = px[mask]
        ppy = py[mask]
        p_std = stds[mask]


        extra = ''
        if confidence != 'low':
            extra = ', '
            extra += {
                'med': '30th',
                'high': '50th',
                'vhigh': '75th',
                'vvhigh': '90th',
            }[confidence]
            extra += ' percentile confidence'
        title = 'Our model'+extra

        fig = plt.figure(figsize=(4, 4),
                         dpi=300,
                         constrained_layout=True)
        # if plot_random:
            # ic('random')
            # ic(len(ppx))
            # alpha = min([0.05 * 72471 / len(ppx), 1.0])
        # else:
            # ic('not random')
            # ic(len(ppx))

    #     alpha = min([0.05 * 8740 / len(ppx), 1.0])
    #     ic(alpha, plot_random, len(ppx))
        alpha = 1.0

        #colors[2, 3]
        main_color = main_color.tolist()
        g = sns.jointplot(ppx, ppy,
                        alpha=alpha,# ax=ax,
                          color=main_color,
    #                     hue=(ppy/p_std)**2,
                        s=0.0,
                        xlim=(3, 10),
                        ylim=(3, 10),
                        marginal_kws=dict(bins=15),
                       )

        ax = g.ax_joint
        snr = (ppy/p_std)**2
        relative_snr = snr / max(snr)
        point_color = relative_snr

        rmse = np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]))**0.5
        snr_rmse = np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]), weights=snr[ppx<8.99])**0.5
        print(f'{confidence} confidence gets RMSE of {rmse:.2f}')
        print(f'Weighted by SNR, this is: {snr_rmse:.2f}')

        # np.save(f'ppx_{args.version}.npy', ppx)
        # np.save(f'ppy_{args.version}.npy', ppy)
        # np.save(f'p_std_{args.version}.npy', p_std)

        ######################################################
        # Bias scores:
        tmpdf = pd.DataFrame({'true': ppx, 'pred': ppy, 'w': snr})
        for lo in range(4, 9):
            hi = lo + 0.99
            considered = tmpdf.query(f'true>{lo} & true<{hi}')
            print(f"Between {lo} and {hi}, the bias is {np.average(considered['pred'] - considered['true']):.3f}",
                    f"and the weighted bias is {np.average(considered['pred'] - considered['true'], weights=considered['w']):.3f}")
        ######################################################

        #Transparency:
        if show_transparency:
            if plot_random:
                transparency_adjuster = 1.0 #0.5 * 0.2
            else:
                transparency_adjuster = 1.0
            point_color = np.concatenate(
                (einops.repeat(colors[2, 3], 'c -> row c', row=len(ppy)),
                 point_color[:, None]*transparency_adjuster), axis=1)
        #color mode:
        else:
            point_color = np.einsum('r,i->ir', main_color, point_color) +\
                np.einsum('r,i->ir', off_color, 1-point_color)



        im = ax.scatter(
                    ppx,
                   ppy, marker='o',
                   c=point_color,
                   s=10,
                   edgecolors='none'
                  )
        ax.plot([4-3, 9+3], [4-3, 9+3], color='k')
        ax.plot([4-3, 9+3], [4+0.61-3, 9+0.61+3], color='k', ls='--')
        ax.plot([4-3, 9+3], [4-0.61-3, 9-0.61+3], color='k', ls='--')
        ax.set_xlim(3+0.9, 10-0.9)
        ax.set_ylim(3+0.9, 10-0.9)
        ax.set_xlabel('Truth')
        ax.set_ylabel('Predicted')
        plt.suptitle(title, y=1.0)
        plt.tight_layout()

        if confidence == 'low':
            plt.savefig(checkpoint_filename + 'comparison.png', dpi=300)
        else:
            plt.savefig(checkpoint_filename + f'_{confidence}_confidence_' + 'comparison.png', dpi=300)

        if logger:
            logger.log_metrics({"comparison": wandb.Image(plt)})

    # +

    mymap = mpl.colors.LinearSegmentedColormap.from_list(
        'mine', [
            [1.0, 1.0, 1.0, 1.0],
            list(point_color[0, :3]) + [1.0]
        ], N=30
    )
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = mymap
    norm = mpl.colors.Normalize(vmin=snr.min(), vmax=snr.max())

    cb1 = mpl.colorbar.ColorbarBase(ax,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('SNR')
    fig.show()
    plt.savefig(checkpoint_filename + 'colorbar.png', dpi=300)

    plt.style.use('default')
    # plt.style.use('science')

    # Idea: KDE plot but different stacked versions showing contours of the residual. Compare with other algorithms.

    palette = sns.color_palette(['#892665', '#B08431', '#294475', '#82A62E'])
    sns.set_palette(palette)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.distplot((py-px)[(px<8.99)], hist=True, kde=True,
                 bins=30, ax=ax,
                 hist_kws={'edgecolor':'black', 'range': [-4, 4]},
                 kde_kws={'linewidth': 4, 'color': 'k'})
    ax.set_xlabel('Residual')
    ax.set_ylabel('Probability')
    ax.set_title('RMS residual under 9: %.3f'% (np.sqrt(np.average(np.abs(py-px)[px<9])),))

    plt.xlim(-3, 3)
    plt.ylim(0, 0.7)


    fig.savefig(checkpoint_filename + 'residual.pdf')

    labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']


    truths.shape#.reshape(-1)

    plt.style.use('default')
    # plt.style.use('science')
    fpr, tpr, _ = roc_curve(y_true=(truths>=9).reshape(-1),
                            y_score=np.average(np.tile(sample_preds, (2, 1, 1))>9, 1).transpose(1, 0).reshape(-1))
    fig = plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, color=colors[0, 3])
    plt.xlabel('fpr')
    plt.ylabel('tpr')

    y_roc = truths > 8.99

    y_score = np.average(sample_preds>= 9, axis=0)
    # ic(y_roc.shape, y_score.shape)

    y_roc = einops.rearrange(y_roc, 'sample run -> (sample run)')
    y_score = einops.repeat(y_score, 'sample -> (sample run)', run=2)

    # ic(y_roc.shape, y_score.shape)
    # # Use median of stds:
    # preds = np.median(_preds[..., 0], 0)
    # stds = np.median(_preds[..., 1], 0)
    snr = np.median(_preds[..., 0], 0)**2 / np.median(_preds[..., 1], 0)**2

    # Use std of samples:
    # snr =  np.average(sample_preds, axis=0)**2/np.std(sample_preds, axis=0)**2
    y_weight = einops.repeat(snr, 'sample -> (sample run)', run=2)

    roc = roc_auc_score(
        y_true=y_roc,
        y_score=y_score,
    )
    weight_roc = roc_auc_score(
        y_true=y_roc,
        y_score=y_score,
        sample_weight=y_weight
    )
    plt.title('AUC ROC = %.3f'%(roc,))

    print(f'Model gets ROC of {roc:.3f}')
    print(f'Model gets weighted ROC of {weight_roc:.3f}')
    # summary_writer.add_figure(
    #     'roc_curve',
    #     fig)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    fig.savefig(checkpoint_filename + 'classification.pdf')

    if logger:
        logger.log_metrics(metrics={'rmse': rmse,
                                    'snr_rmse': snr_rmse,
                                    'roc': roc,
                                    'weighted_roc': weight_roc,})
        logger.log_metrics({"classification": wandb.Image(fig)})


def calc_scores_nonswag(model, train_all=False, logger=None, plot_random=False, use_petit=False, just_rmse=False):
    model.eval()
    model.cuda()

    path = 'plots/' + model.path()
    if use_petit:
        path = 'plots/petit'

    if plot_random:
        path += '_random'

    print('path:', path)

    plt.switch_backend('agg')


    ########################################
    # Stuff with loading colors

    colorstr = """*** Primary color:

       shade 0 = #A0457E = rgb(160, 69,126) = rgba(160, 69,126,1) = rgb0(0.627,0.271,0.494)
       shade 1 = #CD9CBB = rgb(205,156,187) = rgba(205,156,187,1) = rgb0(0.804,0.612,0.733)
       shade 2 = #BC74A1 = rgb(188,116,161) = rgba(188,116,161,1) = rgb0(0.737,0.455,0.631)
       shade 3 = #892665 = rgb(137, 38,101) = rgba(137, 38,101,1) = rgb0(0.537,0.149,0.396)
       shade 4 = #74104F = rgb(116, 16, 79) = rgba(116, 16, 79,1) = rgb0(0.455,0.063,0.31)

    *** Secondary color (1):

       shade 0 = #CDA459 = rgb(205,164, 89) = rgba(205,164, 89,1) = rgb0(0.804,0.643,0.349)
       shade 1 = #FFE9C2 = rgb(255,233,194) = rgba(255,233,194,1) = rgb0(1,0.914,0.761)
       shade 2 = #F1D195 = rgb(241,209,149) = rgba(241,209,149,1) = rgb0(0.945,0.82,0.584)
       shade 3 = #B08431 = rgb(176,132, 49) = rgba(176,132, 49,1) = rgb0(0.69,0.518,0.192)
       shade 4 = #956814 = rgb(149,104, 20) = rgba(149,104, 20,1) = rgb0(0.584,0.408,0.078)

    *** Secondary color (2):

       shade 0 = #425B89 = rgb( 66, 91,137) = rgba( 66, 91,137,1) = rgb0(0.259,0.357,0.537)
       shade 1 = #8C9AB3 = rgb(140,154,179) = rgba(140,154,179,1) = rgb0(0.549,0.604,0.702)
       shade 2 = #697DA0 = rgb(105,125,160) = rgba(105,125,160,1) = rgb0(0.412,0.49,0.627)
       shade 3 = #294475 = rgb( 41, 68,117) = rgba( 41, 68,117,1) = rgb0(0.161,0.267,0.459)
       shade 4 = #163163 = rgb( 22, 49, 99) = rgba( 22, 49, 99,1) = rgb0(0.086,0.192,0.388)

    *** Complement color:

       shade 0 = #A0C153 = rgb(160,193, 83) = rgba(160,193, 83,1) = rgb0(0.627,0.757,0.325)
       shade 1 = #E0F2B7 = rgb(224,242,183) = rgba(224,242,183,1) = rgb0(0.878,0.949,0.718)
       shade 2 = #C9E38C = rgb(201,227,140) = rgba(201,227,140,1) = rgb0(0.788,0.89,0.549)
       shade 3 = #82A62E = rgb(130,166, 46) = rgba(130,166, 46,1) = rgb0(0.51,0.651,0.18)
       shade 4 = #688C13 = rgb(104,140, 19) = rgba(104,140, 19,1) = rgb0(0.408,0.549,0.075)"""

    colors = []
    shade = 0
    for l in colorstr.replace(' ', '').split('\n'):
        elem = l.split('=')
        if len(elem) != 5: continue
        if shade == 0:
            new_color = []
        rgb = lambda x, y, z: np.array([x, y, z]).astype(np.float32)

        new_color.append(eval(elem[2]))

        shade += 1
        if shade == 5:
            colors.append(np.array(new_color))
            shade = 0
    colors = np.array(colors)/255.0

    #################################################3
    ### Load the dataloader


    if use_petit:
        val_dataloader = petit.petit_dataloader(validation=True)

        def sample_model(X_sample):
            return petit.tsurv(X_sample)

    else:
        model.make_dataloaders()
        if plot_random:
            assert model.ssX is not None
            tmp_ssX = copy(model.ssX)
            if train_all:
                model.make_dataloaders(
                    ssX=model.ssX,
                    train=True,
                    plot_random=True)
            else:
                model.make_dataloaders(
                    ssX=model.ssX,
                    train=False,
                    plot_random=True) #train=False means we show the whole dataset (assuming we don't train on it!)

            assert np.all(tmp_ssX.mean_ == model.ssX.mean_)

        val_dataloader = model._val_dataloader

        def sample_model(X_sample):
            # simplified sampling since we don't use SWAG
            return model(X_sample, noisy_val=False)

    # collect the samples

    truths = []
    preds = []
    raw_preds = []

    N = 1  # since we're not using swag, we're deterministic. so just do one sample
    # keeping the rest of the code the same so the shapes dont have to be messed with

    nc = 0
    losses = 0.0
    do_sample = True
    for X_sample, y_sample in tqdm(val_dataloader):
        X_sample = X_sample.cuda()
        y_sample = y_sample.cuda()
        nc += len(y_sample)
        truths.append(y_sample.cpu().detach().numpy())

        raw_preds.append(
            np.array([sample_model(X_sample).cpu().detach().numpy() for _ in range(N)])
        )

    truths = np.concatenate(truths)

    _preds = np.concatenate(raw_preds, axis=1)

    if use_petit:
        sample_preds = _preds
    else:

        # numpy sampling is way too slow:


        def fast_truncnorm(
                loc, scale, left=np.inf, right=np.inf,
                d=10000, nsamp=50, seed=0):
            """Fast truncnorm sampling.

            Assumes scale and loc have the desired shape of output.
            length is number of elements.
            Select nsamp based on expecting at last one sample
                to fit within your (left, right) range.
            Select d based on memory considerations - need to operate on
                a (d, nsamp) array.
            """
            oldscale = scale
            oldloc = loc

            scale = scale.reshape(-1)
            loc = loc.reshape(-1)
            samples = np.zeros_like(scale)
            start = 0

            for start in range(0, scale.shape[0], d):

                end = start + d
                if end > scale.shape[0]:
                    end = scale.shape[0]

                cd = end-start
                rand_out = np.random.randn(
                    nsamp, cd
                )

                rand_out = (
                    rand_out * scale[None, start:end]
                    + loc[None, start:end]
                )

                #rand_out is (nsamp, cd)
                if right == np.inf:
                    mask = (rand_out > left)
                elif left == np.inf:
                    mask = (rand_out < right)
                else:
                    mask = (rand_out > left) & (rand_out < right)

                first_good_val = rand_out[
                    mask.argmax(0), np.arange(cd)
                ]

                samples[start:end] = first_good_val

            return samples.reshape(*oldscale.shape)

        std = _preds[..., 1]
        mean = _preds[..., 0]

        loc = mean
        scale = std

        sample_preds = np.array(
                fast_truncnorm(np.array(mean), np.array(std),
                       left=4, d=874000, nsamp=40));

    if use_petit:
        # sample_preds is a [1, 8720] tensor of means
        # sample with std 1 to create [2000, 8740] tensor of samples
        # samples = np.random.randn(2000, 8740)
        # sample_preds = einops.repeat(sample_preds[0], 'b -> R b', R=2000)
        # sample_preds = sample_preds + samples
        sample_preds = einops.repeat(sample_preds[0], 'b -> R b', R=2000)
        # need to make _preds [1,8740, 2]
        # _preds is currently [1, 8740] of means
        # just use std of one.
        mean = _preds
        std = np.ones_like(mean)
        _preds = einops.rearrange([mean, std], 'two one eight -> one eight two')

    else:

        stable_past_9 = sample_preds >= 9


        _prior = lambda logT: (
            3.27086190404742*np.exp(-0.424033970670719 * logT) -
            10.8793430454878*np.exp(-0.200351029031774 * logT**2)
        )
        normalization = quad(_prior, a=9, b=np.inf)[0]

        prior = lambda logT: _prior(logT)/normalization

        # Let's generate random samples of that prior:
        n_samples = stable_past_9.sum()
        bins = n_samples*4
        top = 100.
        bin_edges = np.linspace(9, top, num=bins)
        cum_values = [0] + list(np.cumsum(prior(bin_edges)*(bin_edges[1] - bin_edges[0]))) + [1]
        bin_edges = [9.] +list(bin_edges)+[top]
        inv_cdf = interp1d(cum_values, bin_edges)
        r = np.random.rand(n_samples)
        samples = inv_cdf(r)

        sample_preds[stable_past_9] = samples

    # # expectation of samples
    # preds = np.average(sample_preds, 0)
    # stds = np.std(sample_preds, 0)

    # # median of samples
    # preds = np.median(sample_preds, 0)
    # stds = (
    #     (lambda x: 0.5*(np.percentile(x, q=50 + 68/2, axis=0) - np.percentile(x, q=50-68/2, axis=0)))
    #     (sample_preds)
    # )

    # # median of dists
    # if use_petit:
        # preds = _preds[
    preds = np.median(_preds[..., 0], 0)
    stds = np.median(_preds[..., 1], 0)

    # # fit a truncated dist using avg, var
    # tmp = fit_trunc_dist.find_mu_sig(sample_preds.T)
    # preds = tmp[:, 0]
    # stds = tmp[:, 1]

    # # with likelihood (slow)
    # tmp = fit_trunc_dist.find_mu_sig_likelihood(sample_preds[:300, :].T)
    # preds = tmp[:, 0]
    # stds = tmp[:, 1]

    # weighted average of mu
    # w_i = 1/_preds[:, :, 1]**2
    # w_i /= np.sum(w_i, 0)
    # preds = np.average(_preds[:, :, 0], 0, weights=w_i)
    # stds = np.average(_preds[:, :, 1]**2, 0)**0.5

    # Check that confidence intervals are satisifed. Calculate mean and std of samples. Take abs(truths - mean)/std = sigma. The CDF of this distrubtion should match that of a Gaussian. Otherwise, rescale "scale".

    tmp_mask = (truths > 6) & (truths < 7) #Take this portion since its far away from truncated parts
    averages = preds#np.average(sample_preds, 0)
    gaussian_stds = stds#np.std(sample_preds, 0)
    sigma = (truths[tmp_mask] - np.tile(averages, (2, 1)).T[tmp_mask])/np.tile(gaussian_stds, (2, 1)).T[tmp_mask]

    np.save(path + 'model_error_distribution.npy', sigma)


    bins = 30
    fig = plt.figure(figsize=(4, 4))
    plt.hist(np.abs(sigma), bins=bins, range=[0, 2.5], density=True,
                color=colors[0, 3],
             alpha=1, label='Model error distribution')
    np.random.seed(0)
    plt.hist(np.abs(np.random.randn(len(sigma))), bins=bins, range=[0, 2.5], density=True,
                color=colors[1, 3],
             alpha=0.5, label='Gaussian distribution')
    plt.ylim(0, 1.2)
    plt.ylabel('Density', fontsize=14)
    plt.xlabel('Error over sigma', fontsize=14)
    # plt.xlabel('$|\mu_θ - y|/\sigma_θ$', fontsize=14)
    plt.legend()
    fig.savefig(path + 'error_dist.pdf')


    # Looks great! We didn't even need to tune it. Just use the same scale as the paper (0.5). Perhaps, however, with epistemic uncertainty, we will need to tune.


    def density_scatter(x, y, xlabel='', ylabel='', clabel='Sample Density', log=False,
        width_mult=1, bins=30, p_cut=None, update_rc=True, ax=None, fig=None, cmap='viridis', **kwargs):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        xy = np.array([x, y]).T
        px = xy[:, 0]
        py = xy[:, 1]
        if p_cut is not None:
            p = p_cut
            range_x = [np.percentile(xy[:, 0], i) for i in [p, 100-p]]
            range_y = [np.percentile(xy[:, 1], i) for i in [p, 100-p]]
            pxy = xy[(xy[:, 0] > range_x[0]) & (xy[:, 0] < range_x[1]) & (xy[:, 1] > range_y[0]) & (xy[:, 1] < range_y[1])]
        else:
            pxy = xy
        px = pxy[:, 0]
        py = pxy[:, 1]
        norm = None
        if log:
            norm = LogNorm()

        h, xedge, yedge, im = ax.hist2d(
            px, py, density=True, norm=norm,
            bins=[int(width_mult*bins), bins], cmap=cmap, **kwargs)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.colorbar(im, ax=ax).set_label(clabel)
        fig.tight_layout()

        return fig, ax

    _preds.shape

    #confidences_to_plot = 'low med high vhigh vvhigh'.split(' ')
    confidences_to_plot = ['low']

    # %matplotlib inline


    show_transparency = True

    main_shade = 3
    main_color = colors[2, main_shade]
    off_color = colors[2, main_shade]


    plt.style.use('default')
    sns.set_style('white')
    plt.rc('font', family='serif')

    # +
    for confidence in confidences_to_plot:
        py = preds

        py = np.clip(py, 4, 9)
        px = np.average(truths, 1)

        mask = np.all(truths < 9.99, 1) # np.all(truths < 8.99, 1)

        if confidence != 'low':
            #tmp_std = np.std(sample_preds, 0)/py
            tmp_std = stds/py
            if confidence == 'high':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 50))
            elif confidence == 'vhigh':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 25))
            elif confidence == 'vvhigh':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 10))
            elif confidence == 'med':
                mask = mask & ((tmp_std) < np.percentile(tmp_std[mask], 70))

        ppx = px[mask]
        ppy = py[mask]
        p_std = stds[mask]


        extra = ''
        if confidence != 'low':
            extra = ', '
            extra += {
                'med': '30th',
                'high': '50th',
                'vhigh': '75th',
                'vvhigh': '90th',
            }[confidence]
            extra += ' percentile confidence'
        title = 'Our model'+extra

        fig = plt.figure(figsize=(4, 4),
                         dpi=300,
                         constrained_layout=True)
        # if plot_random:
            # ic('random')
            # ic(len(ppx))
            # alpha = min([0.05 * 72471 / len(ppx), 1.0])
        # else:
            # ic('not random')
            # ic(len(ppx))

    #     alpha = min([0.05 * 8740 / len(ppx), 1.0])
    #     ic(alpha, plot_random, len(ppx))
        alpha = 1.0

        #colors[2, 3]
        main_color = main_color.tolist()
        g = sns.jointplot(ppx, ppy,
                        alpha=alpha,# ax=ax,
                          color=main_color,
    #                     hue=(ppy/p_std)**2,
                        s=0.0,
                        xlim=(3, 13),
                        ylim=(3, 13),
                        marginal_kws=dict(bins=15),
                       )

        ax = g.ax_joint
        snr = (ppy/p_std)**2
        relative_snr = snr / max(snr)
        point_color = relative_snr
        # pickle.dump([ppx, ppy, p_std], open(path + 'comparison_data.pt', 'wb'))
        # print('Saved pickle to ', path + 'comparison_data.pt')

        rmse = np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]))**0.5
        snr_rmse = np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]), weights=snr[ppx<8.99])**0.5
        print(f'{confidence} confidence gets RMSE of {rmse:.2f}')
        print(f'Weighted by SNR, this is: {snr_rmse:.2f}')
        if just_rmse:
            return rmse

        ######################################################
        # Bias scores:
        tmpdf = pd.DataFrame({'true': ppx, 'pred': ppy, 'w': snr})
        for lo in range(4, 9):
            hi = lo + 0.99
            considered = tmpdf.query(f'true>{lo} & true<{hi}')
            # print(f"Between {lo} and {hi}, the bias is {np.average(considered['pred'] - considered['true']):.3f}",
                    # f"and the weighted bias is {np.average(considered['pred'] - considered['true'], weights=considered['w']):.3f}")
            print(f"Between {lo} and {hi}, the mse is {np.average((considered['pred'] - considered['true'])**2):.3f}",
                    f"and the weighted mse is {np.average((considered['pred'] - considered['true'])**2, weights=considered['w']):.3f}")
        ######################################################

        #Transparency:
        if show_transparency:
            if plot_random:
                transparency_adjuster = 1.0 #0.5 * 0.2
            else:
                transparency_adjuster = 1.0
            point_color = np.concatenate(
                (einops.repeat(colors[2, 3], 'c -> row c', row=len(ppy)),
                 point_color[:, None]*transparency_adjuster), axis=1)
        #color mode:
        else:
            point_color = np.einsum('r,i->ir', main_color, point_color) +\
                np.einsum('r,i->ir', off_color, 1-point_color)



        im = ax.scatter(
                    ppx,
                   ppy, marker='o',
                   c=point_color,
                   s=10,
                   edgecolors='none'
                  )
        ax.plot([4-3, 9+3], [4-3, 9+3], color='k')
        ax.plot([4-3, 9+3], [4+0.61-3, 9+0.61+3], color='k', ls='--')
        ax.plot([4-3, 9+3], [4-0.61-3, 9-0.61+3], color='k', ls='--')
        ax.set_xlim(3+0.9, 10-0.9)
        ax.set_ylim(3+0.9, 10-0.9)

        # ax.set_xlim(3+0.9, 13-0.9)
        # ax.set_ylim(ppy.min(), ppy.max())

        ax.set_xlabel('Truth')
        ax.set_ylabel('Predicted')
        plt.suptitle(title, y=1.0)
        plt.tight_layout()

        if confidence == 'low':
            plt.savefig(path + 'comparison.png', dpi=300)
        else:
            plt.savefig(path + f'_{confidence}_confidence_' + 'comparison.png', dpi=300)

        if logger:
            logger.log_metrics({"comparison": wandb.Image(plt)})

    # +

    mymap = mpl.colors.LinearSegmentedColormap.from_list(
        'mine', [
            [1.0, 1.0, 1.0, 1.0],
            list(point_color[0, :3]) + [1.0]
        ], N=30
    )
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = mymap
    norm = mpl.colors.Normalize(vmin=snr.min(), vmax=snr.max())

    cb1 = mpl.colorbar.ColorbarBase(ax,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('SNR')
    fig.show()
    plt.savefig(path + 'colorbar.png', dpi=300)

    plt.style.use('default')
    # plt.style.use('science')

    # Idea: KDE plot but different stacked versions showing contours of the residual. Compare with other algorithms.

    palette = sns.color_palette(['#892665', '#B08431', '#294475', '#82A62E'])
    sns.set_palette(palette)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.distplot((py-px)[(px<8.99)], hist=True, kde=True,
                 bins=30, ax=ax,
                 hist_kws={'edgecolor':'black', 'range': [-4, 4]},
                 kde_kws={'linewidth': 4, 'color': 'k'})
    ax.set_xlabel('Residual')
    ax.set_ylabel('Probability')
    ax.set_title('RMS residual under 9: %.3f'% (np.sqrt(np.average(np.abs(py-px)[px<9])),))

    plt.xlim(-3, 3)
    plt.ylim(0, 0.7)


    fig.savefig(path + 'residual.pdf')

    labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']


    truths.shape#.reshape(-1)

    plt.style.use('default')
    # plt.style.use('science')
    fpr, tpr, _ = roc_curve(y_true=(truths>=9).reshape(-1),
                            y_score=np.average(np.tile(sample_preds, (2, 1, 1))>9, 1).transpose(1, 0).reshape(-1))
    fig = plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, color=colors[0, 3])
    plt.xlabel('fpr')
    plt.ylabel('tpr')

    y_roc = truths > 8.99

    y_score = np.average(sample_preds>= 9, axis=0)
    # ic(y_roc.shape, y_score.shape)

    y_roc = einops.rearrange(y_roc, 'sample run -> (sample run)')
    y_score = einops.repeat(y_score, 'sample -> (sample run)', run=2)

    # ic(y_roc.shape, y_score.shape)
    # # Use median of stds:
    # preds = np.median(_preds[..., 0], 0)
    # stds = np.median(_preds[..., 1], 0)
    snr = np.median(_preds[..., 0], 0)**2 / np.median(_preds[..., 1], 0)**2

    # Use std of samples:
    # snr =  np.average(sample_preds, axis=0)**2/np.std(sample_preds, axis=0)**2
    y_weight = einops.repeat(snr, 'sample -> (sample run)', run=2)

    roc = roc_auc_score(
        y_true=y_roc,
        y_score=y_score,
    )
    weight_roc = roc_auc_score(
        y_true=y_roc,
        y_score=y_score,
        sample_weight=y_weight
    )
    plt.title('AUC ROC = %.3f'%(roc,))

    print(f'Model gets ROC of {roc:.3f}')
    print(f'Model gets weighted ROC of {weight_roc:.3f}')
    # summary_writer.add_figure(
    #     'roc_curve',
    #     fig)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    fig.savefig(path + 'classification.pdf')

    if logger:
        logger.log_metrics(metrics={'rmse': rmse,
                                    'snr_rmse': snr_rmse,
                                    'roc': roc,
                                    'weighted_roc': weight_roc,})
        logger.log_metrics({"classification": wandb.Image(fig)})

    return rmse, snr_rmse, roc, weight_roc


def calculate_k_results():
    d = {2: {'version': 24880,
             'pysr_version': 11003},
         3: {'version': 74649,
             'pysr_version': 83278},
         4: {'version': 11566,
             'pysr_version': 51254},
         5: {'version': 72646,
             'pysr_version': 55894}}

    overall_results = {}
    for k in d:
        version = d[k]['version']
        pysr_version = d[k]['pysr_version']
        reg = pickle.load(open(f'sr_results/{pysr_version}.pkl', 'rb'))
        results = reg.equations_[0]
        complexities = results['complexity']
        k_results = {}
        for c in complexities:
            args = get_args()
            args.version = version
            args.pysr_version = pysr_version
            args.pysr_model_selection = c
            args.just_rmse = True
            rmse = main(args)
            k_results[c] = rmse
            print(f'k={k}, c={c}, rmse={rmse}')
        overall_results[k] = k_results

    pickle.dump(overall_results, open('k_results2.pkl', 'wb'))


def calculate_f2_lin_results():
    d = {20: 2702,
         10: 13529,
         5: 7307,
         2: 22160}
    for k, version in d.items():
        args = get_args()
        args.version = version
        args.just_rmse = True
        rmse = main(args)
        print(f'k={k}, rmse={rmse}')


def calculate_f1_id_results():
    version = 12370
    pysr_version = 22943

    f1_id_results = {}
    reg = pickle.load(open(f'sr_results/{pysr_version}.pkl', 'rb'))
    results = reg.equations_[0]
    complexities = results['complexity']
    for c in complexities:
        args = get_args()
        args.version = version
        args.pysr_version = pysr_version
        args.pysr_model_selection = c
        args.just_rmse = True
        rmse = main(args)
        f1_id_results[c] = rmse
        print(f'c={c}, rmse={rmse}')

    pickle.dump(results, open('f1_id_results.pkl', 'wb'))


def main(args):
    if args.pysr_version:
        if args.pure_sr:
            model = modules.PureSRNet(args.pysr_version, model_selection=args.pysr_model_selection)
        else:
            model = spock_reg_model.load_with_pysr_f2(version=args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection)
    else:
        model = spock_reg_model.load(version=args.version)

    return calc_scores_nonswag(model, use_petit=args.petit, plot_random=args.plot_random, just_rmse=args.just_rmse)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, default=24880)
    parser.add_argument('--pysr_version', type=int, default=None)
    parser.add_argument('--petit', action='store_true')
    parser.add_argument('--plot_random', action='store_true')
    parser.add_argument('--pure_sr', action='store_true')
    parser.add_argument('--pysr_model_selection', type=str, default='accuracy', help='"best", "accuracy", "score", or an integer of the pysr equation complexity.')
    parser.add_argument('--just_rmse', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)

