#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:38:00 2024

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
half1       = np.zeros_like(gave) #7001 timepoints for -3s -> 4s at 1KHz
half2       = np.zeros_like(gave)

baselined = False
icount    = -1
for i in subs:
    icount += 1
    print(f"\n- - - - working on participant {i} - - - -")
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    stim1locked = mne.epochs.read_epochs(param['s1locked_eyes'], preload=True) #read in data
    times = stim1locked.times
    nonpupchannels = [x for x in stim1locked.ch_names if x != 'pupil']
    
    #get info on %missing data per trial
    trlnans_perc = np.load(file = op.join(trlcheckdir, f"{param['subid']}_stim1locked_eyes_nancheck_perTrial.npy"))
    to_remove = np.where(trlnans_perc>50)[0] #find trials where blanket over 50% of data is missing to remove
    to_keep = trlnans_perc<30
    
    #discard some trials from this vector based on ones already thrown out
    # to_keep = to_keep[stim1locked.metadata.trlid-1]
    stim1locked.metadata = stim1locked.metadata.assign(tokeep = to_keep)
    stim1locked = stim1locked['tokeep == True']
    
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
    
    # tmpgave = np.squeeze(stim1locked.pick(picks='pupil_transformed')._data)
    tmpgave = np.squeeze(stim1locked.copy().drop_channels(nonpupchannels)._data) #if you want the non-transformed pupil size, use this line
    tmpgave_mean = np.nanmean(tmpgave, axis=0)
    gave[icount] = tmpgave_mean
    
    # eyedat = np.squeeze(stim1locked._data)
    eyedat = np.squeeze(stim1locked.copy().drop_channels(nonpupchannels)._data)
    ntrls = eyedat.shape[0]
    # trls = np.random.uniform(0, 1, size=ntrls)<0.5
    trls = np.random.randint(0, 2, size = ntrls) #choose either 0 or 1 randomly
    
    #split into half
    # splits = np.split(eyedat, 2, axis=0) #make into 2 splits
    # half1[icount] = np.nanmean(eyedat[trls==True], axis=0)
    # half2[icount]  = np.nanmean(eyedat[trls==False], axis=0)
    half1[icount]  = np.nanmean(eyedat[trls == 0], axis=0)
    half2[icount]  = np.nanmean(eyedat[trls == 1], axis=0)

#plot each subject, split into two random parts to see if it varies much

fig = plt.figure(figsize=[15,15])
iplot = 0
for isub in range(subs.size):
    iplot += 1
    ax = fig.add_subplot(6,5,iplot)
    ax.set_title(f"participant {subs[isub]}")
    ax.plot(times, half1[isub], color = '#386cb0', lw = 2, label = 'half1')
    ax.plot(times, half2[isub], color = '#fdc086', lw = 2, label = 'half2')
    # ax.axhline(0, lw = 1, ls = 'dashed', color = '#000000')
    ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
    
# there are some ppts whose split halves always look quite different:
    
# on the unbaselined data:
# participants 23, 15, 21, 28, 22, 11, 18, 19

#on the baselined data:
# 18, 34

#v weird shapes: 13, 18, 23, 27, 39

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



    
    
    
    
    
    