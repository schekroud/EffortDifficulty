#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 19:59:07 2024

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
    
    
    # s1corr = stim1locked['rewarded == 1'].pick(picks='pupil_transformed').apply_baseline(baseline).average()
    # s1incorr = stim1locked['unrewarded == 1'].pick(picks='pupil_transformed').apply_baseline(baseline).average()
    
    # corr.append(s1corr)
    # incorr.append(s1incorr)
    
    tmpgave = np.squeeze(stim1locked.pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
    tmpgave_mean = np.nanmean(tmpgave, axis=0)
    gave[icount] = tmpgave_mean
    
    # fig = plt.figure(figsize = [4,3])
    # ax = fig.add_subplot(111)
    # ax.plot(times, tmpgave.T)
    # ax.set_title(f"participant {i}")
    
    tmpdiff2 = np.zeros(shape=[2,times.size]) #correct and incorrect, 7001 time points
    tmpdiff2[0] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="correct" and difficultyOri == 2'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)
    tmpdiff2[1] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="incorrect" and difficultyOri == 2'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)    
    diff2[icount] = tmpdiff2
    
    tmpdiff4 = np.zeros(shape=[2,times.size]) #correct and incorrect, 7001 time points
    tmpdiff4[0] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="correct" and difficultyOri == 4'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)
    tmpdiff4[1] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="incorrect" and difficultyOri == 4'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)    
    diff4[icount] = tmpdiff4
    
    tmpdiff8 = np.zeros(shape=[2,times.size]) #correct and incorrect, 7001 time points
    tmpdiff8[0] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="correct" and difficultyOri == 8'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)
    tmpdiff8[1] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="incorrect" and difficultyOri == 8'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)    
    diff8[icount] = tmpdiff8
    
    tmpdiff12 = np.zeros(shape=[2,times.size]) #correct and incorrect, 7001 time points
    tmpdiff12[0] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="correct" and difficultyOri == 12'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)
    tmpdiff12[1] = np.nanmean(np.squeeze(stim1locked['prevtrlfb=="incorrect" and difficultyOri == 12'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data()), axis=0)    
    diff12[icount] = tmpdiff12
    
    
    tmpcorr = np.squeeze(stim1locked['prevtrlfb=="correct"'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
    tmpcorr = np.nanmean(tmpcorr, axis=0) #average across trials
    
    tmpincorr = np.squeeze(stim1locked['prevtrlfb=="incorrect"'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
    tmpincorr = np.nanmean(tmpincorr, axis=0)
    
    tmpdiff = np.subtract(tmpcorr, tmpincorr)
    
    correct[icount]   = tmpcorr
    incorrect[icount] = tmpincorr
    diff[icount]      = tmpdiff
    
    diffcount = -1
    for theta in thetas: #loop over orientation difficulties
        diffcount +=1
        inds = np.where(stim1locked.metadata.difficultyOri==theta)[0]
        tmp = stim1locked.copy()[inds] #get just trials of this difficulty
        tmpdata = np.squeeze(tmp.pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
        tmpdata = np.nanmean(tmpdata, axis=0)
        difficulties[icount, diffcount] = tmpdata
#%%

diff2_mean,   diff2_sem = np.nanmean(diff2, axis=0), sp.stats.sem(diff2, axis=0, nan_policy='omit', ddof = 0)
diff4_mean,   diff4_sem = np.nanmean(diff4, axis=0), sp.stats.sem(diff4, axis=0, nan_policy='omit', ddof = 0)
diff8_mean,   diff8_sem = np.nanmean(diff8, axis=0), sp.stats.sem(diff8, axis=0, nan_policy='omit', ddof = 0)
diff12_mean, diff12_sem = np.nanmean(diff12, axis=0), sp.stats.sem(diff12, axis=0, nan_policy='omit', ddof = 0)
labels = ['prevtrlcorrect', 'prevtrlincorrect']
cols = ['#756bb1', '#2b8cbe']

diff2_contr  = np.subtract(diff2[:,0,:],  diff2[:,1,:]) #shape: ppts x time
diff4_contr  = np.subtract(diff4[:,0,:],  diff4[:,1,:])
diff8_contr  = np.subtract(diff8[:,0,:],  diff8[:,1,:])
diff12_contr = np.subtract(diff12[:,0,:], diff12[:,1,:])

# plot correct vs incorrect separately for each difficulty level

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(2,2,1)
for acc in range(len(labels)):
    ax.plot(times, diff2_mean[acc], label = labels[acc], color = cols[acc], lw = 1.5)
    ax.fill_between(times,
                    y1 = np.subtract(diff2_mean[acc], diff2_sem[acc]), y2 = np.add(diff2_mean[acc], diff2_sem[acc]),
                    lw = 0, edgecolor = None, color = cols[acc], alpha = 0.3)
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
ax.axhline(0, lw = 1, ls='dashed', color = '#000000')
ax.set_title('difficulty 2')

ax = fig.add_subplot(2,2,2)
for acc in range(len(labels)):
    ax.plot(times, diff4_mean[acc], label = labels[acc], color = cols[acc], lw = 1.5)
    ax.fill_between(times,
                    y1 = np.subtract(diff4_mean[acc], diff4_sem[acc]), y2 = np.add(diff4_mean[acc], diff4_sem[acc]),
                    lw = 0, edgecolor = None, color = cols[acc], alpha = 0.3)
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
ax.axhline(0, lw = 1, ls='dashed', color = '#000000')
ax.set_title('difficulty 4')

ax = fig.add_subplot(2,2,3)
for acc in range(len(labels)):
    ax.plot(times, diff8_mean[acc], label = labels[acc], color = cols[acc], lw = 1.5)
    ax.fill_between(times,
                    y1 = np.subtract(diff8_mean[acc], diff8_sem[acc]), y2 = np.add(diff8_mean[acc], diff8_sem[acc]),
                    lw = 0, edgecolor = None, color = cols[acc], alpha = 0.3)
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
ax.axhline(0, lw = 1, ls='dashed', color = '#000000')
ax.set_title('difficulty 8')

ax = fig.add_subplot(2,2,4)
for acc in range(len(labels)):
    ax.plot(times, diff12_mean[acc], label = labels[acc], color = cols[acc], lw = 1.5)
    ax.fill_between(times,
                    y1 = np.subtract(diff12_mean[acc], diff12_sem[acc]), y2 = np.add(diff12_mean[acc], diff12_sem[acc]),
                    lw = 0, edgecolor = None, color = cols[acc], alpha = 0.3)
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
ax.axhline(0, lw = 1, ls='dashed', color = '#000000')
ax.set_title('difficulty 12')
ax.legend(loc = 'lower left')

fig.savefig(op.join(figdir, 'stim1lockedEyes_DifficultyXPrevtrlcorrectness.pdf'), dpi = 300, format = 'pdf')
fig.savefig(op.join(figdir, 'stim1lockedEyes_DifficultyXPrevtrlcorrectness.png'), dpi = 300, format = 'png')

