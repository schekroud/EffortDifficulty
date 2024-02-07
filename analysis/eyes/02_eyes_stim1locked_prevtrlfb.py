#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:16:49 2024

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
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,     26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now
gave        = np.zeros(shape = [subs.size, 7001])
correct     = np.zeros_like(gave) #7001 timepoints for -3s -> 4s at 1KHz
incorrect   = np.zeros_like(gave)
diff        = np.zeros_like(gave)

baselined=False
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
    to_keep = trlnans_perc<30
    
    #discard some trials from this vector based on ones already thrown out
    # to_keep = to_keep[stim1locked.metadata.trlid-1]
    stim1locked.metadata = stim1locked.metadata.assign(tokeep = to_keep)
    stim1locked = stim1locked['tokeep == True']
    
    
    times = stim1locked.times
    
    #remove some relevant trials?
    if i ==16:
        stim1locked = stim1locked['blocknumber != 4'] #lost the eye in block four of this participant
    if i in [21,25]:
        stim1locked = stim1locked['blocknumber > 1'] #these ppts had problems in block 1 of the task
    
    if i == 26:
        stim1locked = stim1locked['trlid != 192'] #this trial has large amounts of interpolated data, and some nans persist
    stim1locked = stim1locked['fbtrig != 62'] #drop timeout trials
    
    
    #take only trials where they should have realised difficulty has changed (i.e. trials 4 onwards in difficulty sequence)
    stim1locked = stim1locked['diffseqpos >= 4']
    stim1locked = stim1locked['prevtrlfb in ["correct","incorrect"]'] #discard trials where they timed out on the previous trial

    
    baseline = None
    if baselined:
        baseline = [-1, -0.5]
    stim1locked = stim1locked.apply_baseline(baseline)
    
    # s1corr = stim1locked['rewarded == 1'].pick(picks='pupil_transformed').apply_baseline(baseline).average()
    # s1incorr = stim1locked['unrewarded == 1'].pick(picks='pupil_transformed').apply_baseline(baseline).average()
    
    # corr.append(s1corr)
    # incorr.append(s1incorr)
    
    tmpgave = np.squeeze(stim1locked.pick(picks='pupil_transformed')._data)
    tmpgave_mean = np.nanmean(tmpgave, axis=0)
    gave[icount] = tmpgave_mean
    
    # fig = plt.figure(figsize = [4,3])
    # ax = fig.add_subplot(111)
    # ax.plot(times, tmpgave.T)
    # ax.set_title(f"participant {i}")
    
    
    tmpcorr = np.squeeze(stim1locked['prevtrlfb=="correct"'].pick(picks='pupil_transformed')._data)
    tmpcorr = np.nanmean(tmpcorr, axis=0) #average across trials
    
    tmpincorr = np.squeeze(stim1locked['prevtrlfb=="incorrect"'].pick(picks='pupil_transformed')._data)
    tmpincorr = np.nanmean(tmpincorr, axis=0)
    
    tmpdiff = np.subtract(tmpcorr, tmpincorr)
    
    correct[icount]   = tmpcorr
    incorrect[icount] = tmpincorr
    diff[icount]      = tmpdiff
#%%    

#plot individual subject means in case they look weird?


fig = plt.figure(figsize=[15,15])
iplot = 0
for isub in range(subs.size):
    iplot += 1
    ax = fig.add_subplot(6,5,iplot)
    ax.set_title(f"participant {subs[isub]}")
    ax.plot(times, gave[isub], color = '#000000', lw = 2, label = 'gave')
    ax.axhline(0, lw = 1, ls = 'dashed', color = '#000000')
    ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')


#%%

cols = ['#756bb1', '#f03b20']
correct_mean, correct_sem = np.nanmean(correct, axis=0), sp.stats.sem(correct, axis = 0, ddof = 0, nan_policy='omit')
incorrect_mean, incorrect_sem = np.nanmean(incorrect, axis=0), sp.stats.sem(incorrect, axis = 0, ddof = 0, nan_policy='omit')

fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.plot(times, correct_mean, label = 'prevtrl correct', color = cols[0])
ax.fill_between(times, np.subtract(correct_mean, correct_sem), np.add(correct_mean, correct_sem),
                alpha = 0.3, color = cols[0], edgecolor = None, lw = 0)
ax.plot(times, incorrect_mean, label = 'prevtrl incorrect', color = cols[1])
ax.fill_between(times, np.subtract(incorrect_mean, incorrect_sem), np.add(incorrect_mean, incorrect_sem),
                alpha = 0.3, color = cols[1], edgecolor = None, lw = 0)
ax.legend(loc = 'upper right')


#%%

#plot this for each subject? 26 ppts

fig = plt.figure(figsize=[15,15])
iplot = 0
for isub in range(subs.size):
    iplot += 1
    ax = fig.add_subplot(6,5,iplot)
    ax.set_title(f"participant {subs[isub]}")
    ax.plot(times, correct[isub], color = cols[0], lw = 2, label = 'correct')
    ax.plot(times, incorrect[isub], color = cols[1], lw = 2, label = 'incorrect')
    
    
    
    
    
    
    
    
    


