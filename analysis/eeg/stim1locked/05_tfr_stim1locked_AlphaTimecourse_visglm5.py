# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:58:43 2023

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
from funcs import getSubjectInfo, gesd, plot_AR

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
glmnum = 'glm5'
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
betas_cols = ['#000000', '#3182bd', '#756bb1'] #intercept, correctness, trialnumber

copes_mean = copes.copy().mean(axis=0)
copes_sem = sp.stats.sem(copes, axis=0, ddof= 0)
cope_cols = ['#000000', '#3182bd', '#756bb1', '#2ca25f', '#e34a33']

#%%

fig = plt.figure(figsize = (9,9))
plotcount = 0
for ibeta in range(len(regnames)):
    if 'diffseqpos' in regnames[ibeta]:
        plotcount +=1
        ax = fig.add_subplot(4, 3, plotcount)
        ax.plot(times, betas_mean[ibeta], label = regnames[ibeta])
        ax.fill_between(times,
                        y1 = np.add(betas_mean[ibeta], betas_sem[ibeta]),
                        y2 = np.subtract(betas_mean[ibeta], betas_sem[ibeta]),
                        alpha = 0.3, lw = 0)
        ax.set_title(regnames[ibeta])
        ax.set_xlim([-2.8, 3.8])
        ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
        ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
        ax.set_xlabel('time relative to stim onset (s)')
        ax.set_ylabel('Beta (AU)')


fig = plt.figure(figsize = (9,4))
ax = fig.add_subplot(111)
plotcount = 0
for ibeta in range(len(regnames)):
    if 'diffseqpos' in regnames[ibeta]:
        ax.plot(times, betas_mean[ibeta], label = regnames[ibeta])
        ax.fill_between(times,
                        y1 = np.add(betas_mean[ibeta], betas_sem[ibeta]),
                        y2 = np.subtract(betas_mean[ibeta], betas_sem[ibeta]),
                        alpha = 0.3, lw = 0)
        ax.set_title(regnames[ibeta])
        ax.set_xlim([-2.8, 3.8])
        ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
        ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
        ax.set_xlabel('time relative to stim onset (s)')
        ax.set_ylabel('Beta (AU)')
fig.legend(loc = 'lower left')


fig = plt.figure(figsize = (9,4))
ax = fig.add_subplot(111)
plotcount = 0
for ibeta in range(len(regnames)):
    if regnames[ibeta] in ['diffseqpos1', 'diffseqpos2', 'diffseqpos3', 'diffseqpos4', 'diffseqpos5']:
        plotcount +=1
        ax.plot(times, betas_mean[ibeta], label = regnames[ibeta])
        ax.fill_between(times,
                        y1 = np.add(betas_mean[ibeta], betas_sem[ibeta]),
                        y2 = np.subtract(betas_mean[ibeta], betas_sem[ibeta]),
                        alpha = 0.3, lw = 0)
        ax.set_title(regnames[ibeta])
        ax.set_xlim([-2.8, 3.8])
        ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
        ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
        ax.set_xlabel('time relative to stim 1 onset (s)')
        ax.set_ylabel('Beta (AU)')
    ax.legend()
fig.legend(loc = 'lower left')

#%%

fig = plt.figure()
ax = fig.add_subplot(111)
for ibeta in range(len(regnames)):
    ax.plot(times, betas_mean[ibeta],label = regnames[ibeta], c=betas_cols[ibeta])
    ax.fill_between(times,
                    y1 = np.add(betas_mean[ibeta], betas_sem[ibeta]),
                    y2 = np.subtract(betas_mean[ibeta], betas_sem[ibeta]),
                    color = betas_cols[ibeta], alpha = 0.3, lw = 0)
ax.set_xlim([-2.8, 3.8])
ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
ax.set_xlabel('time relative to stim onset (s)')
ax.set_ylabel('Beta (AU)')
fig.suptitle('beta timecourse')
fig.legend()
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_betas'+'.eps'), dpi = 300, format = 'eps')
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_betas'+'.pdf'), dpi = 300, format = 'pdf')


fig = plt.figure()
ax = fig.add_subplot(111)
for icope in range(len(contrasts)):
    ax.plot(times, copes_mean[icope],label = contrasts[icope], c=cope_cols[icope])
    ax.fill_between(times,
                    y1 = np.add(copes_mean[icope], copes_sem[icope]),
                    y2 = np.subtract(copes_mean[icope], copes_sem[icope]),
                    color = cope_cols[icope], alpha = 0.3, lw = 0)
ax.set_xlim([-2.8, 3.8])
ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
ax.set_xlabel('time relative to stim onset (s)')
ax.set_ylabel('Beta (AU)')
fig.suptitle('cope timecourse')
fig.legend()
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_copes'+'.eps'), dpi = 300, format = 'eps')
fig.savefig(fname = op.join(figpath, 'stim1locked_tfr'+glmnum+'_copes'+'.pdf'), dpi = 300, format = 'pdf')




