# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:25:40 2024

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
%matplotlib

sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gauss_smooth

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
datadir = op.join(wd, 'data', 'datafiles', 'combined') #path to single subject datafiles

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,    38, 39])
#drop 36 & 37 as unusable and withdrew from task
smooth    = True #if smoothing single trial alpha timecourse
transform = False #if converting power to decibels (10*log10 power)
twin      = [1,2]
twin2     = [-1.3, -0.3] #get the last second, uncontaminated by decomposition window
for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    
    #this contains all trials
    tfr = mne.time_frequency.read_tfrs(param['stim1locked'].replace('stim1locked', 'stim1locked_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
    
    #this contains just the cleaned trials
    tfr_cleaned = mne.time_frequency.read_tfrs(param['stim1locked'].replace('stim1locked', 'stim1locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5'))[0];
    cleanedtrlid = tfr_cleaned.metadata.trlid.to_numpy()-1
    del(tfr_cleaned)
    tokeep = np.zeros(len(tfr)).astype(int)
    tokeep[cleanedtrlid] = 1
    times = tfr.times
    
    baseline_input = True
    posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8', 'Oz', 'POz']
    
    if baseline_input:
        bline = (-2.7, -2.4) #baseline period for stim1locked

    tfrdat = tfr.copy().apply_baseline(baseline = bline).pick_channels(posterior_channels).data.copy()
    tfrdat = np.mean(tfrdat, axis = 2) #average across the frequency band, results in trials x channels x time
    tfrdat = np.mean(tfrdat, axis = 1) #average across channels now, returns trials x time
    
    if smooth:
        tfrdat = gauss_smooth(tfrdat, sigma = 2)
    if transform:
        tfrdat = np.multiply(10, np.log10(tfrdat))
    twin_times = (times>=twin[0]) & (times <= twin[1]) #true/false for in/outside this time range
    
    #get just data in this time range
    tfrdat = tfrdat[:,twin_times]
    tfrdat = tfrdat.mean(axis=1) #average across time to get alpha power in this time window
    
    tfrs1 = tfrdat.copy()
    tokeep1 = tokeep.copy()
    
    #now get the same info for stim1locked data
    #this contains all trials
    tfr = mne.time_frequency.read_tfrs(param['stim2locked'].replace('stim2locked', 'stim2locked_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
    
    #this contains just the cleaned trials
    tfr_cleaned = mne.time_frequency.read_tfrs(param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5'))[0];
    cleanedtrlid = tfr_cleaned.metadata.trlid.to_numpy()-1
    del(tfr_cleaned)
    tokeep = np.zeros(len(tfr)).astype(int)
    tokeep[cleanedtrlid] = 1
    times = tfr.times
    
    #get baseline values
    bline_vals = np.load(op.join(wd, 'eeg', 's%02d'%i, 'EffortDifficulty_s%02d_stim1locked_baselineValues_AlphaOnly.npy'%i)) #load baseline vals    
    tfrdata = tfr.data.copy()
    #baseline the stim2locked data to the pre-stim1 baseline period
    for itime in range(tfr.times.size): #loop over timepoints
        tfrdata[:,:,:,itime] = np.subtract(tfrdata[:,:,:,itime], bline_vals)
    tfr.data = tfrdata #re-assign the baselined data back into the tfr object
    tfrdat = tfr.copy().pick_channels(posterior_channels).data.copy()
    tfrdat = np.mean(tfrdat, axis = 2) #average across the frequency band, results in trials x channels x time
    tfrdat = np.mean(tfrdat, axis = 1) #average across channels now, returns trials x time
    if smooth:
        tfrdat = gauss_smooth(tfrdat, sigma = 2)
        
    twin_times = (times>=twin2[0]) & (times <= twin2[1]) #true/false for in/outside this time range
    #get just data in this time range
    tfrdat  = tfrdat[:,twin_times]
    tfrdat  = tfrdat.mean(axis=1) #average across time to get alpha power in this time window
    tfrs2   = tfrdat.copy()
    tokeep2 = tokeep.copy()
    
    eegdata = pd.DataFrame(np.array([tfrs1, tokeep1, tfrs2, tokeep2]).T, columns = ['s1eeg', 's1keep', 's2eeg', 's2keep'])
    eegdata = eegdata.assign(trlid = np.arange(len(eegdata))+1,
                             subid = i)
    
    eegdata.to_csv(op.join(datadir, f'{param["subid"]}_eeg_singleTrialMetrics.csv'), index=False)
    