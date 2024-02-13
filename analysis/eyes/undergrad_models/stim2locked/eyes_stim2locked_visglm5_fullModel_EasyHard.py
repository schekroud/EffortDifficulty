#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:42:50 2024

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

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
glmnum = 'glm5'
glmdir = op.join(wd, 'glms', 'eyes', 'stim2locked', 'undergrad_models', glmnum)
figpath = op.join(wd, 'figures', 'eyes', 'stim2locked', 'undergrad_models', glmnum)
if not op.exists(figpath):
    os.mkdir(figpath)

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





subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,         39])
#drop 36 & 37 as unusable and withdrew from task
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,     22, 23, 24,     26, 27, 28,         31, 32, 33, 34, 35,        39]) #subjects from eyetracking analysis oo
nsubs = subs.size

# doesnt matter if you smooth per trial, or smooth the average. 
# for regression, better to smooth trialwise
# good to lightly smooth (1sd gaussian smoothing is not too much) for statistical power
smoothing = True
def gauss_smooth(array, sigma = 10):
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
    ibetas  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedEyes_betas.npy'))
    icopes  = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedEyes_copes.npy'))
    itstats = np.load(file = op.join(glmdir, param['subid'] + '_stim2lockedEyes_tstats.npy'))
    
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

colors = dict(easy = '#2166ac', hard = '#ef8a62',
              easycorrectness = '#2166ac', hardcorrectness = '#ef8a62',
              easyprevcorrectness = '#2166ac', hardprevcorrectness = '#ef8a62',
              hardprevtrlcorr = '#542788', hardprevtrlincorr = '#b2182b',
              easyprevtrlcorr = '#542788', easyprevtrlincorr = '#b2182b',
              hardcorr = '#1a9850', hardincorr = '#b2182b',
              easycorr = '#1a9850', easyincorr = '#b2182b')

#%%
heights = np.round(betas.mean(axis=0).min(axis=1),1) - 0.2e-10
alpha = 0.05
df = nsubs - 1
t_thresh = sp.stats.t.ppf(0.95, df = df)


#plot easy vs hard trials
fig = plt.figure(figsize = [10,4])
ax = fig.add_subplot(121)
for ireg in range(regnames.size):
    regname = regnames[ireg]
    if regname in ['hard', 'easy']:
        ax.plot(times, betas_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(betas_mean[ireg], betas_sem[ireg]), np.subtract(betas_mean[ireg], betas_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)

ax = fig.add_subplot(122)
ax.plot(times, copes_mean[7], lw = 2, color = '#000000', label = 'difference')
ax.fill_between(times, np.add(copes_mean[7], copes_sem[7]), np.subtract(copes_mean[7], copes_sem[7]),
                alpha = 0.3, lw = 0, edgecolor = None, color = '#000000')
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
# fig.tight_layout()
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
# ax.set_xlim([-2.7, 3.7])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)
fig.suptitle('Pupil size ~ current trial performance')
fig.savefig(op.join(figpath, 'model5_accuracy_byDifficultyGrouping.pdf'), format = 'pdf', dpi = 400)

#%%
#plot correct and incorrect trials against each other for each difficulty grouping

fig = plt.figure(figsize = (10,4))
ax = fig.add_subplot(121)
for ireg in range(contrasts.size):
    regname = contrasts[ireg]
    if regname in ['hardcorr', 'hardincorr']:
        ax.plot(times, copes_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(copes_mean[ireg], copes_sem[ireg]), np.subtract(copes_mean[ireg], copes_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)
        
ax = fig.add_subplot(122)
for ireg in range(contrasts.size):
    regname = contrasts[ireg]
    if regname in ['easycorr', 'easyincorr']:
        ax.plot(times, copes_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(copes_mean[ireg], copes_sem[ireg]), np.subtract(copes_mean[ireg], copes_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)

fig.suptitle('Pupil size ~ current trial performance, per difficulty grouping')
# fig.tight_layout()
fig.savefig(op.join(figpath, 'model5_CorrectIncorrect_byDifficulty.pdf'), format = 'pdf', dpi = 400)



#%%


#plot correctness by difficulty grouping
fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
for ireg in range(regnames.size):
    regname = regnames[ireg]
    if regname in ['hardprevcorrectness', 'easyprevcorrectness']:
        ax.plot(times, betas_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(betas_mean[ireg], betas_sem[ireg]), np.subtract(betas_mean[ireg], betas_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
# ax.set_xlim([-2.7, 3.7])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)
fig.suptitle('Pupil size ~ previous trial performance')
fig.savefig(op.join(figpath, 'model5_prevtrlfbAcc_byDifficultyGrouping.pdf'), format = 'pdf', dpi = 400)

#%%
fig = plt.figure(figsize = (10,4))
ax = fig.add_subplot(121)
for ireg in range(contrasts.size):
    regname = contrasts[ireg]
    if regname in ['hardprevtrlcorr', 'hardprevtrlincorr']:
        ax.plot(times, copes_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(copes_mean[ireg], copes_sem[ireg]), np.subtract(copes_mean[ireg], copes_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)
        
ax = fig.add_subplot(122)
for ireg in range(contrasts.size):
    regname = contrasts[ireg]
    if regname in ['easyprevtrlcorr', 'easyprevtrlincorr']:
        ax.plot(times, copes_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(copes_mean[ireg], copes_sem[ireg]), np.subtract(copes_mean[ireg], copes_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False)

fig.suptitle('Pupil size ~ previous trial performance, per difficulty grouping')
fig.tight_layout()
# fig.savefig(op.join(figpath, 'model5_PrevtrlCorrectIncorrect_byDifficulty.pdf'), format = 'pdf', dpi = 400)
