#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:29:37 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import statsmodels as sm
import statsmodels.api as sma
import os.path as op
from matplotlib import pyplot as plt
import mne
import os
import sys
#%matplotlib
mne.viz.set_browser_backend('qt')

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gauss_smooth
from eyefuncs_mne import find_blinks

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

eyedir  = op.join(wd, 'data', 'eyes', 'asc')
datadir = op.join(wd, 'data', 'datafiles', 'combined') #path to single subject datafiles
figdir  = op.join(wd, 'figures', 'eyes') 
trlcheckdir = op.join(wd, 'data', 'eyes', 'trialchecks', 'stim1locked')
trlcheckdir2 = op.join(wd, 'data', 'eyes', 'trialchecks', 'stim2locked')

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
# subs = [10]
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now

chan2use = 'pupil'
twin1a = [0.5,2]   #time window for stim1locked pupil size
twin1b = [2,4]
twin2a  = [-2,0] #time window for stim2locked pupil size
twin2b  = [0,0.9] #time window for stim2locked pupil size

for i in subs:
    print(f"\n- - - - working on participant {i} - - - -")
    sub      = dict(loc = 'laptop', id = i)
    param    = getSubjectInfo(sub)
    
    epoched  = mne.epochs.read_epochs(param['s1locked_eyes'], preload = True)
    nonchans = [x for x in epoched.ch_names if x != chan2use]
    epoched.drop_channels(nonchans)
    times = epoched.times
    
    #get info on %missing data per trial
    trlnans_perc = np.load(file = op.join(trlcheckdir, f"{param['subid']}_stim1locked_eyes_nancheck_perTrial.npy"))
    to_remove    = np.where(trlnans_perc>50)[0] #find trials where blanket over 50% of data is missing to remove
    to_keep      = trlnans_perc<30
    baseline     = [-1, -0.5]
    epoched      = epoched.apply_baseline(baseline) #baseline the data if there is a baseline to be applied
    
    #smooth the data first
    eyedat = np.squeeze(epoched._data)
    eyedat = gauss_smooth(eyedat, sigma = 50)
    epoched._data = eyedat.reshape([eyedat.shape[0], 1, eyedat.shape[1]])
    
    eyedat_twin1a       = np.squeeze(epoched.copy().crop(tmin = twin1a[0], tmax = twin1a[1])._data)
    eyedat_twin1a       = np.nanmean(eyedat_twin1a, axis = 1) #average across time to get single baseline value for this trial
    eyedat_win1a        = eyedat_twin1a.copy()

    #for time window 
    eyedat_twin1b       = np.squeeze(epoched.copy().crop(tmin = twin1b[0], tmax = twin1b[1])._data)
    eyedat_twin1b       = np.nanmean(eyedat_twin1b, axis = 1) #average across time to get single baseline value for this trial
    eyedat_win1b        = eyedat_twin1b.copy()
    tokeep1      = to_keep.copy()
    # np.save(file = op.join(eyedir, f'{param["subid"]}_eyes_stim1lockedPupil_twin_{twin[0]}s_{twin[1]}s.npy'), arr = eyedat1)
    
    #stim2locked metric
    epoched2  = mne.epochs.read_epochs(param['s2locked_eyes'], preload = True)
    nonchans = [x for x in epoched2.ch_names if x != chan2use]
    epoched2.drop_channels(nonchans)
    times = epoched2.times
    
    #get info on %missing data per trial
    trlnans_perc = np.load(file = op.join(trlcheckdir2, f"{param['subid']}_stim2locked_eyes_nancheck_perTrial.npy"))
    to_remove    = np.where(trlnans_perc>50)[0] #find trials where blanket over 50% of data is missing to remove
    to_keep      = trlnans_perc<30
    
    eyedat = np.squeeze(epoched2._data.copy())
    blinevals = np.load(op.join(eyedir, '%s_eyes_stim1locked_baselinevalues.npy'%(param['subid'])))
    for itime in range(times.size):
        eyedat[:,itime] = np.subtract(eyedat[:,itime], blinevals)
    #smooth the data
    eyedat = gauss_smooth(eyedat, sigma = 50)
    eyedat = eyedat.reshape([eyedat.shape[0], 1, eyedat.shape[1]])
    epoched2._data = eyedat
    
    epoched2_twin2a = np.squeeze(epoched2.copy().crop(tmin = twin2a[0], tmax = twin2a[1])._data)
    epoched2_twin2a = np.nanmean(epoched2_twin2a, axis = 1) #average across time to get single baseline value for this trial
    eyedat_twin2a = epoched2_twin2a.copy()
    tokeep2 = to_keep.copy()
    
    epoched2_twin2b = np.squeeze(epoched2.copy().crop(tmin = twin2b[0], tmax = twin2b[1])._data)
    epoched2_twin2b = np.nanmean(epoched2_twin2b, axis = 1) #average across time to get single baseline value for this trial
    eyedat_twin2b   = epoched2_twin2b.copy()
    # np.save(file = op.join(eyedir, f'{param["subid"]}_eyes_stim1lockedPupil_twin_{twin[0]}s_{twin[1]}s.npy'), arr = eyedat2)
    
    #collate together into a dataframe
    edata = pd.DataFrame(np.array([eyedat_win1a, eyedat_win1b, tokeep1, eyedat_twin2a, eyedat_twin2b, tokeep2]).T,
                         columns = ['s1pupil_twin1', 's1pupil_twin2', 's1keep', 's2pupil_twin1', 's2pupil_twin2', 's2keep'])
    edata = edata.assign(trlid = np.arange(len(edata))+1,
                         subid = i)
    
    edata.to_csv(op.join(datadir, f'{param["subid"]}_pupil_singleTrialMetrics.csv'), index=False)
    