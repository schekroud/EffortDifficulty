#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:30:14 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import stats
%matplotlib
mne.viz.set_browser_backend('qt')

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, clusterperm_test

#set some standard figure parameters
#print(plt.style.available) #see available themes
# plt.style.use('ggplot')
plt.style.use('seaborn-v0_8-whitegrid') #this sets a default white background, grey ticklines
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['axes.titlesize']   = 16
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['figure.frameon']   = False
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['axes.labelsize']   = 14
plt.rcParams['savefig.edgecolor'] = '#FFFFFF'
plt.rcParams['savefig.facecolor'] = '#FFFFFF'

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
glmnum = 'glm2'
glmdir = op.join(wd, 'glms', 'eyes', 'stim2locked', 'undergrad_models', glmnum)
figpath = op.join(wd, 'figures', 'eyes', 'stim2locked', 'undergrad_models', glmnum)
if not op.exists(figpath):
    os.mkdir(figpath)

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,         39])
#drop 36 & 37 as unusable and withdrew from task
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,     22, 23, 24,     26, 27, 28,         31, 32, 33, 34, 35,        39]) #subjects from eyetracking analysis oo
nsubs = subs.size

times = np.round(np.load(op.join(glmdir, 'glm_timerange.npy')), decimals = 2)
regnames = np.load(op.join(glmdir, 'regressor_names.npy'))
contrasts = np.load(op.join(glmdir, 'contrast_names.npy'))

for iglm in [0, 1]:
    addtopath = ''
    if iglm == 1:
        addtopath = '_nodiff2'
    betas = np.empty(shape = (nsubs, len(regnames), len(times)))
    copes = np.empty(shape = (nsubs, len(contrasts), len(times)))
    tstats = np.empty(shape = (nsubs, len(contrasts), len(times)))
    
    count = -1
    for i in subs:
        count += 1
        print('loading subject %02d'%(i)+ '  -  (%02d/%02d)'%(count+1, nsubs))
        sub   = dict(loc = 'workstation', id = i)
        param = getSubjectInfo(sub)
        ibetas  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedEyes_betas%s.npy'%addtopath))
        icopes  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedEyes_copes%s.npy'%addtopath))
        itstats = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedEyes_tstats%s.npy'%addtopath))
        
        betas[count] = ibetas
        copes[count] = icopes
        tstats[count] = itstats
    betas_mean, betas_sem = betas.mean(0), sp.stats.sem(betas, axis =0, ddof = 0, nan_policy = 'omit')
    copes_mean, copes_sem = copes.mean(0), sp.stats.sem(copes, axis =0, ddof = 0, nan_policy = 'omit')
    
    colors = dict(intercept = '#000000', prevtrlcorrectness = '#2166ac', trialnumber = '#8c510a', correct = '#542788', incorrect = '#b2182b')
    
    
    #visualise correct and incorrect trials, and their difference (2 subplots)
    heights = np.round(copes.mean(axis=0).min(axis=1),1) - 20
    alpha = 0.05
    df = nsubs - 1
    t_thresh = sp.stats.t.ppf(0.95, df = df); #t_thresh = None
    tmin, tmax = -2, 0
    
    
    fig = plt.figure(figsize = [9, 3])
    ax = fig.add_subplot(121)
    ax.axvline(0.5, color = '#696969', lw = 1, ls = 'dashed')
    for icope in range(contrasts.size):
        cname = contrasts[icope]
        if cname in ['correct', 'incorrect']:
            ax.plot(times, copes_mean[icope], lw = 2, color = colors[cname], label = 'prevtrl'+cname)
            ax.fill_between(times, np.add(copes_mean[icope], copes_sem[icope]), np.subtract(copes_mean[icope], copes_sem[icope]),
                            alpha = 0.3, lw = 0, edgecolor = None, color = colors[cname])
    ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    ax.set_xlabel('time relative to stim2 onset (s)')
    ax.set_ylabel('Pupil size (Beta, AU)')
    ax.legend(loc = 'lower left', frameon = False)
    
    ax = fig.add_subplot(122)
    ax.axvline(0.5, color = '#696969', lw = 1, ls = 'dashed')
    for icope in range(contrasts.size):
        cname = contrasts[icope]
        if cname in ['prevtrlcorrectness']:
            ax.plot(times, copes_mean[icope], lw = 2, color = colors[cname], label = cname)
            ax.fill_between(times, np.add(copes_mean[icope], copes_sem[icope]), np.subtract(copes_mean[icope], copes_sem[icope]),
                            alpha = 0.3, lw = 0, edgecolor = None, color = colors[cname])
            
            # run cluster permutation testing
            # (used in deciding what clusters to visualise in figure)
            t, clu, clupv, _ = clusterperm_test(data = copes,
                                                labels = contrasts,
                                                of_interest = cname,
                                                times = times,
                                                tmin = tmin, tmax = tmax, 
                                                out_type = 'indices', n_permutations = 100000,
                                                threshold = t_thresh,
                                                tail = 1,
                                                n_jobs = 4)
            clu = [x[0] for x in clu]
            print('clusters for '+cname + ' - ' +str(clupv))
            
            times_twin = times[np.where(np.logical_and(np.greater_equal(times, tmin), np.less_equal(times, tmax)))[0]]    
            masks_cope = np.asarray(clu, dtype='object')[clupv <= alpha]
            for mask in masks_cope:
                itmin = times_twin[mask[0]]
                itmax = times_twin[mask[-1]]
                ax.hlines(y = heights[icope],
                          xmin = itmin,
                          xmax = itmax,
                          lw = 3, color = colors[cname], alpha = 1)
    ax.axvspan(xmin = tmin, xmax = tmax, alpha = 0.3, color = '#bdbdbd', lw = 0, edgecolor = None)
    ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    ax.set_xlabel('time relative to stim2 onset (s)')
    ax.set_ylabel('Pupil size (Beta, AU)')
    fig.tight_layout()
    fig.savefig(op.join(figpath, 'model2_allTrials_prevtrlfb_prevtrlCorrectIncorrect%s.pdf'%addtopath), format = 'pdf', dpi = 400)