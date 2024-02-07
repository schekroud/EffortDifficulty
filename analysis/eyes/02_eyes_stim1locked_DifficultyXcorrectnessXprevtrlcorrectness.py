#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:28:37 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import os.path as op
from matplotlib import pyplot as plt
import mne
import os
import sys
%matplotlib
mne.viz.set_browser_backend('qt')
np.set_printoptions(suppress=True)

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo
from eyefuncs_mne import find_blinks, transform_pupil

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

eyedir  = op.join(wd, 'data', 'eyes', 'asc')
datadir = op.join(wd, 'data', 'datafiles', 'combined') #path to single subject datafiles
figdir  = op.join(wd, 'figures', 'eyes') 
trlcheckdir = op.join(wd, 'data', 'eyes', 'trialchecks', 'stim1locked')

drop_gaze = True

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
# subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35,             39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now
gave        = np.zeros(shape = [subs.size, 7001])
correct     = np.zeros_like(gave) #7001 timepoints for -3s -> 4s at 1KHz
incorrect   = np.zeros_like(gave)
diff        = np.zeros_like(gave)

thetas = [2,4,8,12]
difficulties = np.zeros(shape = [subs.size, len(thetas), 7001])
diff2 = np.zeros(shape = (subs.size, 2, 7001))
diff4 = np.zeros_like(diff2)
diff8 = np.zeros_like(diff2)
diff12 = np.zeros_like(diff2)

diffs = [2, 4, 8, 12]
corrs = [0, 1]
prevtrlcorrs = ['correct', 'incorrect']

monster = np.empty(shape = [subs.size, len(diffs), len(corrs), len(prevtrlcorrs), 7001])

icount = -1
for i in subs:
    icount += 1
    print(f"\n- - - - working on participant {i} - - - -")
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    stim1locked = mne.epochs.read_epochs(param['s1locked_eyes'], preload=True) #read in data
    
    #get info on %missing data per trial
    trlnans_perc = np.load(file = op.join(trlcheckdir, f"{param['subid']}_stim1locked_eyes_nancheck_perTrial.npy"))
    to_remove = np.where(trlnans_perc>50)[0] #find trials where blanket over 50% of data is missing to remove
    
    times = stim1locked.times
    
    #remove some relevant trials?
    if i ==16:
        stim1locked = stim1locked['blocknumber != 4'] #lost the eye in block four of this participant
    if i in [21,25]:
        stim1locked = stim1locked['blocknumber > 1'] #these ppts had problems in block 1 of the task
    
    if i == 26:
        stim1locked = stim1locked['trlid != 192'] #this trial has large amounts of interpolated data, and some nans persist
    stim1locked = stim1locked['fbtrig != 62'] #drop timeout trials
    
    baseline = None
    # baseline = [-0.2, 0]

    tmpgave = np.squeeze(stim1locked.copy().pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
    tmpgave_mean = np.nanmean(tmpgave, axis=0)
    gave[icount] = tmpgave_mean

    
    diffs = [2, 4, 8, 12]
    corrs = [0, 1]
    prevtrlcorrs = ['correct', 'incorrect']
    
    tmp = np.empty(shape = [len(diffs), len(corrs), len(prevtrlcorrs), times.size])
    diffcount = -1
    acccount  = -1
    for idiff in diffs:
        diffcount +=1
        acccount  = -1
        for acc in corrs:
            acccount +=1
            prevtrlcount = -1
            for prevtrl in prevtrlcorrs:
                prevtrlcount +=1
                inds = np.where(stim1locked.metadata.difficultyOri.eq(idiff) & 
                                stim1locked.metadata.PerceptDecCorrect.eq(acc) & 
                                stim1locked.metadata.prevtrlfb.eq(prevtrl))[0]
                tmpdata = stim1locked[inds].copy()
                tmppdata = np.nanmean(np.squeeze(tmpdata.copy().pick('pupil_transformed').apply_baseline(baseline)._data), axis=0)
                tmp[diffcount, acccount, prevtrlcount] = tmppdata
    monster[icount] = tmp
#%%

monster_mean = np.nanmean(monster, axis=0)
monster_sem  = sp.stats.sem(monster, axis=0, ddof = 0, nan_policy = 'omit')

labels = ['incorr_corr', 'incorr_incorr', 'corr_corr', 'corr_incorr']
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']


fig = plt.figure(figsize = [8,8])
for idiff in range(len(diffs)):
    ax = fig.add_subplot(2, 2, idiff+1)
    tmp, tmp2 = monster_mean[idiff], monster_sem[idiff]
    incorr_corr = ax.plot(times, tmp[0,0], label = labels[0], color = colors[0], lw = 1)
    ax.fill_between(times,
                   y1 = np.add(     tmp[0,0], tmp2[0,0]),
                   y2 = np.subtract(tmp[0,0], tmp2[0,0]),
                   lw = 0, edgecolor = None, color = colors[0], alpha = 0.3)
    
    incorr_incorr = ax.plot(times, tmp[0,1], label = labels[1], color = colors[1], lw = 1)
    ax.fill_between(times,
                   y1 = np.add(     tmp[0,1], tmp2[0,1]),
                   y2 = np.subtract(tmp[0,1], tmp2[0,1]),
                   lw = 0, edgecolor = None, color = colors[1], alpha = 0.3)
    
    corr_corr = ax.plot(times, tmp[1,0], label = labels[2], color = colors[2], lw = 1)
    ax.fill_between(times,
                   y1 = np.add(     tmp[1,0], tmp2[1,0]),
                   y2 = np.subtract(tmp[1,0], tmp2[1,0]),
                   lw = 0, edgecolor = None, color = colors[2], alpha = 0.3)
    
    corr_incorr = ax.plot(times, tmp[1,1], label = labels[3], color = colors[3], lw = 1)
    ax.fill_between(times,
                   y1 = np.add(     tmp[1,1], tmp2[1,1]),
                   y2 = np.subtract(tmp[1,1], tmp2[1,1]),
                   lw = 0, edgecolor = None, color = colors[3], alpha = 0.3)
    # ax.legend(loc='upper center')
    ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
    ax.axhline(0, lw = 1, ls='dashed', color = '#000000')
fig.legend(handles = [incorr_corr, incorr_incorr, corr_corr, corr_incorr],labels = labels, loc = 'lower left', ncols=4)
    
    














