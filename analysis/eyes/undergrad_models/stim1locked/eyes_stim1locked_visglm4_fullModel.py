#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 18:45:34 2024

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
glmnum = 'glm4'
glmdir = op.join(wd, 'glms', 'eyes', 'stim1locked', 'undergrad_models', glmnum)
figpath = op.join(wd, 'figures', 'eyes', 'stim1locked', 'undergrad_models', glmnum)
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

iglm = 0
addtopath = ''
if iglm == 1:
    addtopath = '_nodiff2'

times = np.round(np.load(op.join(glmdir, 'glm_timerange%s.npy'%addtopath)), decimals = 2)
regnames = np.load(op.join(glmdir, 'regressor_names%s.npy'%addtopath))
contrasts = np.load(op.join(glmdir, 'contrast_names%s.npy'%addtopath))

betas = np.empty(shape = (nsubs, len(regnames), len(times)))
copes = np.empty(shape = (nsubs, len(contrasts), len(times)))
tstats = np.empty(shape = (nsubs, len(contrasts), len(times)))

count = -1
for i in subs:
    count += 1
    print('loading subject %02d'%(i)+ '  -  (%02d/%02d)'%(count+1, nsubs))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    ibetas  = np.load(file = op.join(glmdir, param['subid'] + '_stim1lockedEyes_betas%s.npy'%addtopath))
    icopes  = np.load(file = op.join(glmdir, param['subid'] + '_stim1lockedEyes_copes%s.npy'%addtopath))
    itstats = np.load(file = op.join(glmdir, param['subid'] + '_stim1lockedEyes_tstats%s.npy'%addtopath))
    
    betas[count] = ibetas
    copes[count] = icopes
    tstats[count] = itstats
betas_mean, betas_sem = betas.mean(0), sp.stats.sem(betas, axis =0, ddof = 0, nan_policy = 'omit')
copes_mean, copes_sem = copes.mean(0), sp.stats.sem(copes, axis =0, ddof = 0, nan_policy = 'omit')

colors = dict(diff2 = '#e41a1c', diff4 = '#fc8d62', diff8 = '#41ab5d', diff12 = '#006d2c',
              d2correctness = '#e41a1c', d4correctness = '#fc8d62', d8correctness = '#41ab5d', d12correctness = '#006d2c',
              d2prevcorrectness = '#e41a1c', d4prevcorrectness = '#fc8d62', d8prevcorrectness = '#41ab5d', d12prevcorrectness = '#006d2c',
              correct = '#1a9850', incorrect = '#b2182b')

#%%

heights = np.round(betas.mean(axis=0).min(axis=1),1) - 0.2e-10
alpha = 0.05
df = nsubs - 1
t_thresh = sp.stats.t.ppf(0.95, df = df)

labels = ['diff2', 'diff4', 'diff8', 'diff12']
#plot prevtrl correctness by difficulties
fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
regcount = -1
for ireg in range(regnames.size):
    regname = regnames[ireg]
    if regname in ['d2correctness', 'd4correctness', 'd8correctness', 'd12correctness']:
        print(regcount)
        regcount += 1 
        ilabel = labels[regcount]
        ax.plot(times, betas_mean[ireg], lw = 2, color = colors[regname], label = regname)
        ax.fill_between(times, np.add(betas_mean[ireg], betas_sem[ireg]), np.subtract(betas_mean[ireg], betas_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
# ax.set_xlim([-2.7, 3.7])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 2)
ax.set_title('current trial performance contrast ~ difficulty')
fig.tight_layout()
fig.savefig(op.join(figpath, 'model4_accuracy_byDifficulty.pdf'), format = 'pdf', dpi = 400)

#%%
#plot correct and incorrect for each difficulty, separately in subplots
# np.array([np.arange(contrasts.size), contrasts]).T    # this lets you see the indices for certain contrasts more easily

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(221)
ax.set_title('difficulty2')
ax.plot(times, copes_mean[13], label = 'correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[13], copes_sem[13]), np.subtract(copes_mean[13], copes_sem[13]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[14], label = 'incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[14], copes_sem[14]), np.subtract(copes_mean[14], copes_sem[14]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)

ax = fig.add_subplot(222)
ax.set_title('difficulty4')
ax.plot(times, copes_mean[15], label = 'correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[15], copes_sem[15]), np.subtract(copes_mean[15], copes_sem[15]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[16], label = 'incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[16], copes_sem[16]), np.subtract(copes_mean[16], copes_sem[16]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)

ax = fig.add_subplot(223)
ax.set_title('difficulty8')
ax.plot(times, copes_mean[17], label = 'correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[17], copes_sem[17]), np.subtract(copes_mean[17], copes_sem[17]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[18], label = 'incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[18], copes_sem[18]), np.subtract(copes_mean[18], copes_sem[18]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)

ax = fig.add_subplot(224)
ax.set_title('difficulty12')
ax.plot(times, copes_mean[19], label = 'correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[19], copes_sem[19]), np.subtract(copes_mean[19], copes_sem[19]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[20], label = 'incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[20], copes_sem[20]), np.subtract(copes_mean[20], copes_sem[20]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)
fig.suptitle('pupil size ~ current trial performance')
fig.tight_layout()
fig.savefig(op.join(figpath, 'model4_CorrectIncorrectTrials_byDifficulty.pdf'), format = 'pdf', dpi = 400)



#%%
heights = np.round(betas.mean(axis=0).min(axis=1),1) - 0.2e-10
alpha = 0.05
df = nsubs - 1
t_thresh = sp.stats.t.ppf(0.95, df = df)

labels = ['diff2', 'diff4', 'diff8', 'diff12']
#plot prevtrl correctness by difficulties
fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
regcount = -1
for ireg in range(regnames.size):
    regname = regnames[ireg]
    if regname in ['d2prevcorrectness', 'd4prevcorrectness', 'd8prevcorrectness', 'd12prevcorrectness']:
        regcount +=1 
        ilabel = labels[regcount]
        ax.plot(times, betas_mean[ireg], lw = 2, color = colors[regname], label = ilabel)
        ax.fill_between(times, np.add(betas_mean[ireg], betas_sem[ireg]), np.subtract(betas_mean[ireg], betas_sem[ireg]),
                        alpha = 0.3, lw = 0, edgecolor = None, color = colors[regname])
ax.set_xlim([-2.7, 3.7])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 2)
ax.set_title('previous trial correctness contrast ~ difficulty')
fig.tight_layout()
fig.savefig(op.join(figpath, 'model4_prevtrlcorrectness_byDifficulty.pdf'), format = 'pdf', dpi = 400)
# 
#%%

#plot pupil size for previously correct and incorrect trials separately, separately for each difficulty level
# np.array([np.arange(contrasts.size), contrasts]).T    # this lets you see the indices for certain contrasts more easily

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(221)
ax.set_title('difficulty2')
ax.plot(times, copes_mean[21], label = 'previous correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[21], copes_sem[21]), np.subtract(copes_mean[21], copes_sem[21]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[22], label = 'previous incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[22], copes_sem[22]), np.subtract(copes_mean[22], copes_sem[22]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)

ax = fig.add_subplot(222)
ax.set_title('difficulty4')
ax.plot(times, copes_mean[23], label = 'previous correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[23], copes_sem[23]), np.subtract(copes_mean[23], copes_sem[23]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[24], label = 'previous incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[24], copes_sem[24]), np.subtract(copes_mean[24], copes_sem[24]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)

ax = fig.add_subplot(223)
ax.set_title('difficulty8')
ax.plot(times, copes_mean[25], label = 'previous correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[25], copes_sem[25]), np.subtract(copes_mean[25], copes_sem[25]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[26], label = 'previous incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[26], copes_sem[26]), np.subtract(copes_mean[26], copes_sem[26]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)

ax = fig.add_subplot(224)
ax.set_title('difficulty12')
ax.plot(times, copes_mean[27], label = 'previous correct', color = colors['correct'], lw = 2)
ax.fill_between(times, np.add(copes_mean[27], copes_sem[27]), np.subtract(copes_mean[27], copes_sem[27]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['correct'])
ax.plot(times, copes_mean[28], label = 'previous incorrect', color = colors['incorrect'], lw = 2)
ax.fill_between(times, np.add(copes_mean[28], copes_sem[28]), np.subtract(copes_mean[28], copes_sem[28]),
                alpha = 0.3, edgecolor = None, lw = 0, color = colors['incorrect'])
ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('Pupil size (Beta, AU)')
ax.legend(loc = 'lower left', frameon = False, ncols = 1)
fig.suptitle('pupil size ~ previous trial performance')
fig.tight_layout()
fig.savefig(op.join(figpath, 'model4_PreviousCorrectIncorrectTrials_byDifficulty.pdf'), format = 'pdf', dpi = 400)



