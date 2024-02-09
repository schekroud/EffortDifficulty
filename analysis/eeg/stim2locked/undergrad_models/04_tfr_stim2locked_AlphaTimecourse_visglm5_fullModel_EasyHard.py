# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:04:11 2024

@author: sammirc
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

sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, clusterperm_test

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
glmnum = 'glm5'
glmdir = op.join(wd, 'glms', 'stim2locked', 'alpha_timecourses', 'undergrad_models', glmnum)
figpath = op.join(wd, 'figures', 'eeg_figs', 'stim2locked', 'undergrad_models', glmnum)
if not op.exists(figpath):
    os.mkdir(figpath)

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,    38, 39])
#drop 36 & 37 as unusable and withdrew from task
# subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,     22, 23, 24,     26, 27, 28,         31, 32, 33, 34, 35,        39]) #subjects from eyetracking analysis oo
nsubs = subs.size

# doesnt matter if you smooth per trial, or smooth the average. 
# for regression, better to smooth trialwise
# good to lightly smooth (1sd gaussian smoothing is not too much) for statistical power
smoothing = True
def gauss_smooth(array, sigma = 2):
    return sp.ndimage.gaussian_filter1d(array, sigma=sigma, axis = 1) #smooths across time, given 2d array of trials x time


times = np.round(np.load(op.join(glmdir, 'glm_timerange.npy')), decimals = 2)
regnames = np.load(op.join(glmdir, 'regressor_names.npy'))
contrasts = np.load(op.join(glmdir, 'contrast_names.npy'))

betas = np.empty(shape = (nsubs, len(regnames), len(times)))
copes = np.empty(shape = (nsubs, len(contrasts), len(times)))
tstats = np.empty(shape = (nsubs, len(contrasts), len(times)))

count = -1
for i in subs:
    count += 1
    print('loading subject %02d'%(i)+ '  -  (%02d/%02d)'%(count+1, nsubs))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    ibetas  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_betas.npy'))
    icopes  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_copes.npy'))
    itstats = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_tstats.npy'))
    
    betas[count] = ibetas
    copes[count] = icopes
    tstats[count] = itstats

if smoothing:
    #smooth the single subject beta/cope timecourses, rather than single trial timecourses
    for i in range(nsubs):
        # for ibeta in range(len(regnames)):
        betas[i] = gauss_smooth(betas[i].copy())
        
        # for icope in range(len(contrasts)):
        copes[i] = gauss_smooth(copes[i].copy())

betas_mean, betas_sem = betas.mean(0), sp.stats.sem(betas, axis =0, ddof = 0, nan_policy = 'omit')
copes_mean, copes_sem = copes.mean(0), sp.stats.sem(copes, axis =0, ddof = 0, nan_policy = 'omit')
#%%
# colors = dict(intercept = '#000000', prevtrlcorrectness = '#2166ac', trialnumber = '#8c510a', correct = '#542788', incorrect = '#b2182b')

colors = dict(diff2 = '#e41a1c', diff4 = '#fc8d62', diff8 = '#b2df8a', diff12 = '#33a02c',
              d2correctness = '#e41a1c', d4correctness = '#fc8d62', d8correctness = '#b2df8a', d12correctness = '#33a02c',
              d2prevcorrectness = '#e41a1c', d4prevcorrectness = '#fc8d62', d8prevcorrectness = '#b2df8a', d12prevcorrectness = '#33a02c')

colors = dict(easy = '#2166ac', hard = '#ef8a62',
              easycorrectness = '#2166ac', hardcorrectness = '#ef8a62',
              easyprevcorrectness = '#2166ac', hardprevcorrectness = '#ef8a62')

#%%
heights = np.round(betas.mean(axis=0).min(axis=1),1) - 0.2e-10
alpha = 0.05
df = nsubs - 1
t_thresh = sp.stats.t.ppf(0.95, df = df)


#plot easy vs hard trials
fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
for ireg in range(regnames.size):
    regname = regnames[ireg]
    if regname in ['hard', 'easy']:
        ax.plot(times, betas_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(betas_mean[ireg], betas_sem[ireg]), np.subtract(betas_mean[ireg], betas_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.set_xlim([-2.7, 1.7])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Alpha power (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)
fig.savefig(op.join(figpath, 'model5_easyHard.pdf'), format = 'pdf', dpi = 400)

#%%

#plot correctness by difficulty grouping
fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
for ireg in range(regnames.size):
    regname = regnames[ireg]
    if regname in ['hardcorrectness', 'easycorrectness']:
        ax.plot(times, betas_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(betas_mean[ireg], betas_sem[ireg]), np.subtract(betas_mean[ireg], betas_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.set_xlim([-2.7, 1.7])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Alpha power (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)
fig.savefig(op.join(figpath, 'model5_accuracy_byDifficultyGrouping.pdf'), format = 'pdf', dpi = 400)

#%%

heights = np.round(betas.mean(axis=0).min(axis=1),1) - 0.2e-10
alpha = 0.05
df = nsubs - 1
t_thresh = sp.stats.t.ppf(0.95, df = df)

tmin, tmax = -2, 0


#plot prevtrl correctness by difficulties
fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.axvspan(xmin = tmin, xmax = tmax, color = '#bdbdbd', alpha = 0.3, edgecolor = None, lw = 0)
for ireg in range(regnames.size):
    regname = regnames[ireg]
    if regname in ['hardprevcorrectness', 'easyprevcorrectness']:
        ax.plot(times, betas_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(betas_mean[ireg], betas_sem[ireg]), np.subtract(betas_mean[ireg], betas_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
        
        #(used in deciding what clusters to visualise in figure)
        # tmin, tmax = times.min(), times.max() #times.max() #timerange for cluster perm test
        t, clu, clupv, _ = clusterperm_test(data = betas,
                                            labels = regnames,
                                            of_interest = regname,
                                            times = times,
                                            tmin = tmin, tmax = tmax, 
                                            out_type = 'indices', n_permutations = 100000,
                                            threshold = t_thresh,
                                            tail = 1,
                                            n_jobs = 3)
        clu = [x[0] for x in clu]
        print('clusters for '+regname + ' - ' +str(clupv))
        
        times_twin = times[np.where(np.logical_and(np.greater_equal(times, tmin), np.less_equal(times, tmax)))[0]]    
        masks_cope = np.asarray(clu)[clupv <= alpha]
        for mask in masks_cope:
            itmin = times_twin[mask[0]]
            itmax = times_twin[mask[-1]]
            ax.hlines(y = heights[ireg],
                      xmin = itmin,
                      xmax = itmax,
                      lw = 3, color = colors[regname], alpha = 1)
            
ax.set_xlim([-2.7, 1.7])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Alpha power (Beta, AU)')
ax.legend(loc = 'lower right', frameon = False, ncols = 1)
fig.savefig(op.join(figpath, 'model4_prevtrlcorrectness_byDifficulty.pdf'), format = 'pdf', dpi = 400)