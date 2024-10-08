#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:15:12 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import os.path as op
import statsmodels as sm
import statsmodels.api as sma
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

subs   = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs   = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
subs   = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35,             39])
# subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,     26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
#29/30 - 29 falling asleep throughout, 30 separated into two recordings
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now

#based on noise in the across-trial pupil data, there are some participants with more noise than others:
# there are two ways of handling this:
    #1 - exclude trials where there is over some % missing (e.g. if missing 50% of data, ignore trial)
    #2 - exclude participants who look excessively noisy across trials
#based on option 2, may want to exclude the following:
    #13 (like 2 noisy blocks),
    #14 (systematic blinker @ 1s after stim onset)
    #16 (missing an entire block of task),
    #19 (noisy throughout)
    #20 (noisy throughout)
    #29 noisy throughout
# subs = np.array([10, 11, 12,    15,    17, 18,        21, 22, 23, 24, 25, 26, 27, 28,    30, 31, 32, 33, 34, 35,       39]) #21 remain

gave        = np.zeros(shape = [subs.size, 7001])
correct     = np.zeros_like(gave) #7001 timepoints for -3s -> 4s at 1KHz
incorrect   = np.zeros_like(gave)
diff        = np.zeros_like(gave)

thetas = [2,4,8,12]
difficulties = np.zeros(shape = [subs.size, len(thetas), 7001])

baselined        = False 
regout_prevtrlfb = True
use_transformed  = True
chan2use = 'pupil'


icount = -1
for i in subs:
    icount += 1
    print(f"\n- - - - working on participant {i} - - - -")
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    stim1locked = mne.epochs.read_epochs(param['s1locked_eyes'], preload=True) #read in data
    
    #get zscored trial numbr to include as regressor for time on task (in case it affects alpha estimated)
    tmpdf = pd.DataFrame(np.array([np.add(np.arange(320),1).astype(int), 
                          sp.stats.zscore(np.add(np.arange(320),1))]).T, columns = ['trlid', 'trlid_z'])
    
    tmpdf = tmpdf.query('trlid in @stim1locked.metadata.trlid').reset_index(drop=True)
    tmpdf2 = stim1locked.metadata.copy().reset_index(drop=True)
    tmpdf2 = tmpdf2.assign(trlidz = tmpdf.trlid_z)
    stim1locked.metadata = tmpdf2
    
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
    stim1locked = stim1locked['prevtrlfb in ["correct", "incorrect"]']
    
    #take only trials where they should have realised difficulty has changed (i.e. trials 4 onwards in difficulty sequence)
    stim1locked = stim1locked['diffseqpos >= 4']
    
    baseline = None
    if baselined:
        # baseline = [-1, -0.5]
        baseline = [-0.5, 0]
    
    stim1locked = stim1locked.apply_baseline(baseline) #baseline the data if there is a baseline to be applied
    
    nonchans = [x for x in stim1locked.ch_names if x != chan2use]    
    eyedat = np.squeeze(stim1locked.drop_channels(nonchans)._data) #get data
    bdata  = stim1locked.metadata.copy()
    usedat = eyedat.copy() #specify for now we are using the data as a whole
    
    cleaned = np.zeros_like(eyedat)
    if regout_prevtrlfb:
        print(f"\nregressing out some parts of the task (previous trial feedback, trial number, etc)")
        #set up model with intercept and previous trial feedback, take residuals forwards
        for tp in range(times.size):
            pupil = eyedat[:,tp].copy()
            
            add_intercept = True
            #set up design matrix
            # intercept = np.ones_like(pupil)
            prevtrlfb = bdata.prevtrlfb.to_numpy()
            prevtrlfb = sp.stats.zscore(np.where(prevtrlfb == 'correct',1, -1))
            trlid     = bdata.trlidz.to_numpy() #zscored trial number for time on task
            # accuracy  = bdata.PerceptDecCorrect.to_numpy()
            # accuracy  = sp.stats.zscore(np.where(accuracy == 1, 1, -1))
            dm = pd.DataFrame(np.array([prevtrlfb, trlid]).T, columns = ['prevtrlfb', 'trlid'])
            if add_intercept:
                dm = sma.add_constant(dm)
            
            tplm = sm.regression.linear_model.OLS(pupil, dm, missing = 'none').fit()
            cleaned[:,tp] = tplm.resid
        
        usedat = cleaned.copy() #if we want to regress out things, then we need to be using residuals from the model
    
    
    #if you regress out the intercept, the grand mean for the participant is zero ...
    # tmpgave_mean = np.nanmean(usedat, axis=0)
    # gave[icount] = tmpgave_mean
    # fig = plt.figure(figsize = [4,3])
    # ax = fig.add_subplot(111)
    # ax.plot(times, usedat.T)
    # ax.set_title(f"participant {i}")
    # ax.plot(times, usedat.mean(0), color = '#000000', lw = 2)
    
    for itheta in range(len(thetas)):
        idiffinds = np.where(bdata.difficultyOri == thetas[itheta])[0]
        tmp = np.squeeze(usedat[idiffinds]) #get data for this difficulty
        difficulties[icount, itheta] = np.nanmean(tmp, axis=0) #average across trials
    
    # fig = plt.figure(figsize = [4,3])
    # ax = fig.add_subplot(111)
    # ax.plot(times, difficulties[icount].T, label = ['diff2', 'diff4', 'diff8', 'diff12'])
    # ax.set_title(f"participant {i}")
    # fig.legend()
    
    #get correct and incorrect trials separately too
    corrinds = np.where(bdata.rewarded == 1)[0]
    incorrinds = np.where(bdata.unrewarded == 1)[0]
    tmpcorr   = np.nanmean(usedat[corrinds].copy(),   axis = 0)
    tmpincorr = np.nanmean(usedat[incorrinds].copy(), axis = 0)
    correct[icount]   = tmpcorr
    incorrect[icount] = tmpincorr
    diff[icount]      = np.subtract(tmpcorr, tmpincorr)
#%%
#visualise difficulties

diffs_mean = np.nanmean(difficulties, axis=0)
diffs_sem  = sp.stats.sem(difficulties, axis=0, ddof = 0, nan_policy = 'omit')
diffcols = ['#e41a1c', '#fc8d62', '#b2df8a', '#33a02c']
difflabels = ['difficulty 2', 'difficulty 4', 'difficulty 8', 'difficulty 12']

fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)

for itheta in range(len(thetas)):
    tmpmean, tmpsem = diffs_mean[itheta], diffs_sem[itheta]
    ax.plot(times, tmpmean, color = diffcols[itheta], label = difflabels[itheta], lw = 1)
    ax.fill_between(times, y1 = np.subtract(tmpmean, tmpsem), y2 = np.add(tmpmean, tmpsem),
                    color = diffcols[itheta], edgecolor = None, alpha = 0.3)
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
# ax.axhline(0, lw = 1, ls='dashed', color = '#000000')
if baselined:
    ax.axvspan(xmin = baseline[0], xmax = baseline[1], alpha = 0.3, color = '#bdbdbd', edgecolor = None, lw = 0)
ax.set_xlabel('time rel to stim1 onset (s)')
ax.set_ylabel('Pupil size (AU)')
ax.legend(loc = 'lower left')#, bbox_to_anchor=(1,0.7))


# fig = plt.figure(figsize = [6,4])
# ax = fig.add_subplot(111)
# labels = ['correct', 'incorrect']
# cols = ['#31a354', '#f03b20']
# corr_mean, corr_sem     = np.nanmean(correct, axis=0), sp.stats.sem(correct, axis = 0, ddof = 0, nan_policy = 'omit')
# incorr_mean, incorr_sem = np.nanmean(incorrect, axis=0), sp.stats.sem(incorrect, axis = 0, ddof = 0, nan_policy = 'omit')

# ax.plot(times, corr_mean, color = cols[0], label = 'correct', lw = 1)
# ax.fill_between(times, np.add(corr_mean, corr_sem), np.subtract(corr_mean, corr_sem),
#                 color = cols[0], alpha = 0.3, edgecolor = None, lw = 0)
# ax.plot(times, incorr_mean, color = cols[1], label = 'correct', lw = 1)
# ax.fill_between(times, np.add(incorr_mean, incorr_sem), np.subtract(incorr_mean, incorr_sem),
#                 color = cols[1], alpha = 0.3, edgecolor = None, lw = 0)



#%%
#per difficulty, plot each subject in case there is one really skewing things for difficulty 4?
fig = plt.figure(figsize = (9, 9))
for itheta in range(len(thetas)):
    tmpdifficulty = difficulties[:,itheta,:]
    ax = fig.add_subplot(2, 2, itheta+1)
    ax.plot(times, tmpdifficulty.T)
    ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
    # ax.axhline(0, lw = 1, ls='dashed', color = '#000000')
    ax.set_xlabel('time rel to stim1 onset (s)')
    ax.set_ylabel('pupil (AU)')
    ax.set_title(f"difficulty {thetas[itheta]}")
    if baselined:
        ax.axvspan(xmin = baseline[0], xmax = baseline[1], alpha = 0.3, color = '#bdbdbd', edgecolor = None, lw = 0)
    # ax.set_ylim([-10, 10])

#%%
# plot just difficulty 4, per subject
diff4 = difficulties[:,1,:].copy() #get just difficulty 4 data
fig = plt.figure(figsize = [15,15])
plotcount = 0
for isub in range(subs.size):
    # tmpdat = difficulties[isub]
    plotcount += 1
    ax = fig.add_subplot(5,5, plotcount)
    ax.set_title(f"participant {subs[isub]}")
    ax.plot(times, diff4[isub], lw = 2)
    ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    # ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    if baselined:
        ax.axvspan(xmin = baseline[0], xmax = baseline[1], alpha = 0.3, color = '#bdbdbd', edgecolor = None, lw = 0)
    
#%%

#for each subject separately, plot pupil for each difficulty
diffcols = ['#e41a1c', '#fc8d62', '#b2df8a', '#33a02c']
difflabels = ['difficulty 2', 'difficulty 4', 'difficulty 8', 'difficulty 12']
#plot each difficulty level, separately for each subject
fig = plt.figure(figsize = [15,15])
plotcount = 0
for isub in range(subs.size):
    tmpdat = difficulties[isub]
    plotcount += 1
    ax = fig.add_subplot(5,5, plotcount)
    ax.set_title(f"participant {subs[isub]}")
    ax.plot(times, tmpdat[0], color = diffcols[0], lw = 2)
    ax.plot(times, tmpdat[1], color = diffcols[1], lw = 2)
    ax.plot(times, tmpdat[2], color = diffcols[2], lw = 2)
    ax.plot(times, tmpdat[3], color = diffcols[3], lw = 2)
    ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    # ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    if baselined:
        ax.axvspan(xmin = baseline[0], xmax = baseline[1], alpha = 0.3, color = '#bdbdbd', edgecolor = None, lw = 0)

#%%

#group this as easy (8 12) vs hard (2 4) trials

#for each subject separately, plot pupil for each difficulty grouping
diffcols = ['#e41a1c', '#fc8d62', '#b2df8a', '#33a02c']
difflabels = ['difficulty 2', 'difficulty 4', 'difficulty 8', 'difficulty 12']
#plot each difficulty level, separately for each subject
fig = plt.figure(figsize = [15,15])
plotcount = 0
for isub in range(subs.size):
    tmpdat = difficulties[isub]
    hard = tmpdat.copy()[:2,:].mean(axis=0)
    easy = tmpdat.copy()[2:,:].mean(axis=0)    
    plotcount += 1
    ax = fig.add_subplot(5,5, plotcount)
    ax.set_title(f"participant {subs[isub]}")
    ax.plot(times, hard, color = diffcols[0], lw = 2, label = 'hard')
    ax.plot(times, easy, color = diffcols[3], lw = 2, label = 'easy')
    ax.legend(loc = 'lower left')
    ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    # ax.axhline(0, ls = 'dashed', color = '#000000', lw = 1.5)
    if baselined:
        ax.axvspan(xmin = baseline[0], xmax = baseline[1], alpha = 0.3, color = '#bdbdbd', edgecolor = None, lw = 0)


#%%    
weirdsubs = np.where((tmpdifficulty <-6).sum(axis=1))[0]
subs[weirdsubs] #participants 18 and 25 are a bit weird

for isub in weirdsubs:
    #gave, incorrect, correct, diff, difficulties
    tmpgave = gave[isub]
    tmpcorr = correct[isub]
    tmpincorr = incorrect[isub]
    tmpdiffs = difficulties[isub]
    
    fig = plt.figure(figsize = [10,4])
    ax = fig.add_subplot(121)
    ax.plot(times, tmpgave, color = 'k', label = 'gave')
    ax.plot(times, tmpcorr, color = 'g', label = 'correct')
    ax.plot(times, tmpincorr, color = 'r', label = 'incorrect')
    ax.legend(loc = 'lower left', ncols = 3)
    
    ax = fig.add_subplot(122)
    for itheta in range(len(thetas)):
        ax.plot(times, tmpdiffs[itheta], label = difflabels[itheta], color = diffcols[itheta])
    ax.legend(loc = 'right', bbox_to_anchor = (1.3,0.5))
    fig.suptitle(f"participant {subs[isub]}")