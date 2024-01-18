# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:09:19 2023

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

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')

from funcs import getSubjectInfo, gesd, plot_AR

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)


glmnum = 'glm3'
glmdir = op.join(wd, 'glms', 'stim1locked', 'alpha_timecourses', glmnum)
figpath = op.join(wd, 'figures', 'eeg_figs', 'stim1locked', glmnum)


if not op.exists(figpath):
    os.mkdir(figpath)
glmdir = op.join(wd, 'glms', 'stim1locked', 'alpha_timecourses', glmnum)

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
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
    ibetas  = np.load(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_betas_.npy'))
    icopes  = np.load(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_copes_.npy'))
    itstats = np.load(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_tstats_.npy'))
    
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
#%%
            
betas_mean = betas.copy().mean(axis=0)
betas_sem = sp.stats.sem(betas, axis = 0, ddof=0)
betas_cols = ['#2ca25f', '#e34a33','#2ca25f', '#e34a33','#2ca25f', '#e34a33','#2ca25f', '#e34a33', '#756bb1'] #corr/incorr for 4 difficulties, then trialnumber
# betas_cols = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#756bb1'] #diff2, diff4, diff8, diff12, trialnumber

copes_mean = copes.copy().mean(axis=0)
copes_sem = sp.stats.sem(copes, axis=0, ddof= 0)
cope_cols = ['#2ca25f', '#e34a33','#2ca25f', '#e34a33','#2ca25f', '#e34a33','#2ca25f', '#e34a33', '#756bb1', '#000000','#d7191c', '#fdae61', '#a6d96a', '#1a9641'] #diff2, diff4, diff8, diff12, trialnumber, grandmean
#%%
difficulties = ['difficulty 2', 'difficulty 4', 'difficulty 6', 'difficulty 8']
fig = plt.figure(figsize = (10, 6))
plotcount = 0
for iplot in np.arange(8):
    if np.mod(np.add(1,np.arange(8)),2)[iplot] == 1:
        plotcount += 1
        ax = fig.add_subplot(2, 2, plotcount)
        ax.set_xlim([-2.8, 3.8])
        ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
        ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
        ax.set_xlabel('time relative to stim1 onset (s)')
        ax.set_ylabel('Alpha power (Beta, AU)')
        ax.set_title(difficulties[plotcount-1])
    ax.plot(times, betas_mean[iplot], label = regnames[iplot], c = betas_cols[iplot])
    ax.fill_between(times,
                    y1 = np.add(betas_mean[iplot], betas_sem[iplot]),
                    y2 = np.subtract(betas_mean[iplot], betas_sem[iplot]),
                    color = betas_cols[iplot], alpha = 0.3, lw = 0)
    ax.legend(loc = 'lower left')
fig.tight_layout()
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_betas'+'.eps'), dpi = 300, format = 'eps')
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_betas'+'.pdf'), dpi = 300, format = 'pdf')



#%% plot copes showing posterior alpha suppression for each difficulty level

fig = plt.figure()
ax = fig.add_subplot(111)
for iplot in np.arange(10, len(contrasts)):
    ax.plot(times, copes_mean[iplot],label = contrasts[iplot], c=cope_cols[iplot])
    ax.fill_between(times,
                    y1 = np.add(copes_mean[iplot], copes_sem[iplot]),
                    y2 = np.subtract(copes_mean[iplot], copes_sem[iplot]),
                    color = cope_cols[iplot], alpha = 0.3, lw = 0)
ax.set_xlim([-2.8, 3.8])
ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
ax.set_xlabel('time relative to stim onset (s)')
ax.set_ylabel('Beta (AU)')
fig.suptitle('cope timecourse')
fig.legend()
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_copes'+'.eps'), dpi = 300, format = 'eps')
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_copes'+'.pdf'), dpi = 300, format = 'pdf')