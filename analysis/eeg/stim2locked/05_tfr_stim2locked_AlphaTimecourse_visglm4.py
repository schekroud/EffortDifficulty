# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:10:09 2023

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


glmnum = 'glm4'
glmdir = op.join(wd, 'glms', 'stim2locked', 'alpha_timecourses', glmnum)
figpath = op.join(wd, 'figures', 'eeg_figs', 'stim2locked', glmnum)


if not op.exists(figpath):
    os.mkdir(figpath)
glmdir = op.join(wd, 'glms', 'stim2locked', 'alpha_timecourses', glmnum)

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,28, 29, 30, 31, 32, 33, 34])
# subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,     22, 23, 24,     26])
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


times = np.load(op.join(glmdir, 'glm_timerange.npy'))
regnames = np.load(op.join(glmdir, 'regressor_names.npy'))
contrasts = np.load(op.join(glmdir, 'contrast_names.npy'))

difficulties = [2, 4, 8, 12]
betas  = np.empty(shape = (nsubs, len(difficulties), regnames.size,  times.size))
copes  = np.empty(shape = (nsubs, len(difficulties), contrasts.size, times.size))
tstats = np.empty(shape = (nsubs, len(difficulties), contrasts.size, times.size))

count = -1
for i in subs:
    count +=1 #to help storing data in structures
    diffcount = -1
    print('loading subject %02d'%(i)+ '  -  (%02d/%02d)'%(count+1, nsubs))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    for diff in difficulties:
        diffcount +=1        
        ibetas  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_betas_difficulty%s.npy'%str(diff)))
        icopes  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_copes_difficulty%s.npy'%str(diff)))
        itstats = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_tstats_difficulty%s.npy'%str(diff)))
        
        betas[count, diffcount] = ibetas
        copes[count, diffcount] = icopes
        tstats[count, diffcount] = itstats

if smoothing:
    #smooth the single subject beta/cope timecourses, rather than single trial timecourses
    for i in range(nsubs):
        for idiff in range(len(difficulties)):
            betas[i, idiff] = gauss_smooth(betas[i, idiff].copy())
            copes[i, idiff] = gauss_smooth(copes[i, idiff].copy())
#%%
            
betas_mean = betas.copy().mean(axis=0)
betas_sem = sp.stats.sem(betas, axis = 0, ddof=0)
betas_cols = ['#000000', '#3182bd', '#756bb1'] #intercept, correctness, trialnumber (for each difficulty level)

copes_mean = copes.copy().mean(axis=0)
copes_sem = sp.stats.sem(copes, axis=0, ddof= 0)
cope_cols = ['#000000', '#3182bd', '#756bb1', '#2ca25f', '#e34a33'] #intercept, correctness, trialnumber, correct, incorrect
#%%

fig = plt.figure(figsize = (10,7))
for diff in range(len(difficulties)):
    idiff = difficulties[diff]
    ax = fig.add_subplot(2,2, diff+1)
    ax.set_title('difficulty '+str(idiff))
    for ireg in range(len(regnames)):
        if ireg >= 0:
            ax.plot(times, betas_mean[diff, ireg].T, label = regnames[ireg], c = betas_cols[ireg])
            ax.fill_between(times,
                            y1 = np.add(betas_mean[diff, ireg], betas_sem[diff, ireg]),
                            y2 = np.subtract(betas_mean[diff, ireg], betas_sem[diff, ireg]),
                            color = betas_cols[ireg], alpha = 0.3, lw = 0)
    ax.set_xlim([-2.7, 1.7])
    ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
    ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
    ax.set_xlabel('time relative to stim 2 onset (s)')
    ax.set_ylabel('Beta (AU)')
    ax.legend()
fig.tight_layout()
fig.savefig(fname = op.join(figpath, 'stim2locked_tfr'+glmnum+'_betas'+'.eps'), dpi = 300, format = 'eps')
fig.savefig(fname = op.join(figpath, 'stim2locked_tfr'+glmnum+'_betas'+'.pdf'), dpi = 300, format = 'pdf')


fig = plt.figure(figsize = (10,7))
for diff in range(len(difficulties)):
    idiff = difficulties[diff]
    ax = fig.add_subplot(2,2, diff+1)
    ax.set_title('difficulty '+str(idiff))
    for icope in range(len(contrasts)):
        if contrasts[icope] in ['correctness', 'correct', 'incorrect']:
            ax.plot(times, copes_mean[diff, icope].T, label = contrasts[icope], c = cope_cols[icope])
            ax.fill_between(times,
                            y1 = np.add(copes_mean[diff, icope], copes_sem[diff, icope]),
                            y2 = np.subtract(copes_mean[diff, icope], copes_sem[diff, icope]),
                            color = cope_cols[icope], alpha = 0.3, lw = 0)
    ax.set_xlim([-2.7, 1.7])
    ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
    ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
    ax.set_xlabel('time relative to stim 2 onset (s)')
    ax.set_ylabel('Beta (AU)')
    ax.legend()
fig.tight_layout()
fig.savefig(fname = op.join(figpath, 'stim2locked_tfr'+glmnum+'_copes'+'.eps'), dpi = 300, format = 'eps')
fig.savefig(fname = op.join(figpath, 'stim2locked_tfr'+glmnum+'_copes'+'.pdf'), dpi = 300, format = 'pdf')
#%%

diffcols =['#d7191c', '#fdae61', '#a6d96a', '#1a9641'] 
difflabels = ['difficulty 2', 'difficulty 4', 'difficulty 8', 'difficulty 12']
fig = plt.figure()
ax = fig.add_subplot(111)
for idiff in range(len(difficulties)):
    ax.plot(times, betas_mean[idiff, 0], c = diffcols[idiff], label = difflabels[idiff])
    ax.fill_between(times,
                    y1 = np.add(betas_mean[idiff,0], betas_sem[idiff, 0]),
                    y2 = np.subtract(betas_mean[idiff,0], betas_sem[idiff, 0]),
                    color = diffcols[idiff], alpha = 0.3, lw = 0)
ax.set_xlim([-2.7, 1.7])
ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
ax.axvline(x=0, ls = 'dashed', color = '#000000', lw = 1)
ax.set_xlabel('time relative to stim 2 onset (s)')
ax.set_ylabel('Beta (AU)')
ax.legend()
fig.savefig(fname = op.join(figpath, 'stim2locked_tfr'+glmnum+'_betas_difficulty'+'.eps'), dpi = 300, format = 'eps')
fig.savefig(fname = op.join(figpath, 'stim2locked_tfr'+glmnum+'_betas_difficulty'+'.pdf'), dpi = 300, format = 'pdf')
