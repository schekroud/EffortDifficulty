#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:15:55 2023

@author: sammi
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
import glmtools as glm
%matplotlib
mne.viz.set_browser_backend('qt')


# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')

from funcs import getSubjectInfo

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)

subs = np.array([10, 11, 12, 13, 14, 15, 16])

def gauss_smooth(array, sigma = 2):
    return sp.ndimage.gaussian_filter1d(array, sigma=sigma, axis = 1) #smooths across time, given 2d array of trials x time


#%%

transform   = False #make into 10log10 power (dB)
baseline    = True

gmean     = np.empty(shape = [subs.size, 600])
correct   = np.empty(shape = [subs.size, 600])
incorrect = np.empty(shape = [subs.size, 600])

d2   = np.empty(shape = [subs.size, 600])
d4   = np.empty(shape = [subs.size, 600])
d8   = np.empty(shape = [subs.size, 600])
d12  = np.empty(shape = [subs.size, 600])

count = -1 #loop over subjects
for i in subs:
    count +=1 #add 1 to be able t add data
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    
    tfr = mne.time_frequency.read_tfrs(param['stim1locked'].replace('stim1locked', 'stim1locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
    # tfr = mne.time_frequency.read_tfrs(param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
    
    #comes with metadata attached            
    tfr = tfr['fbtrig != 62'] #drop timeout trials
    #reduce down to single timecourse of alpha in posterior channels
    posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8']
    
    #set baseline params to test stuff
    if baseline:
        bline = (None, None)
        bline = (-2.7, -2.4) #baseline period for stim1locked
    elif not baseline:
        bline = None
        
    tfrdat = tfr.copy().apply_baseline(baseline = bline).pick_channels(posterior_channels).data.copy()
    tfrdat = np.mean(tfrdat, axis = 2) #average across the frequency band, results in trials x channels x time
    tfrdat = np.mean(tfrdat, axis = 1) #average across channels now, returns trials x time
    tfrdat = gauss_smooth(tfrdat, sigma = 2)
    
    if transform:
        tfrdat = np.multiply(10, np.log10(tfrdat))
    
    times = tfr.times
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(times, np.mean(tfrdat, axis=0))
    # fig.suptitle('grandmean alpha')
    
    corr = tfr.metadata.rewarded.to_numpy()
    incorr = tfr.metadata.unrewarded.to_numpy()
    difficulty = tfr.metadata.difficultyOri.to_numpy()
    
    diff2 = np.where(difficulty  == 2)[0]
    diff4 = np.where(difficulty  == 4)[0]
    diff8 = np.where(difficulty  == 8)[0]
    diff12 = np.where(difficulty == 12)[0]

    corrdat = tfrdat[corr,:]
    incorrdat = tfrdat[incorr,:]
    
    diff2dat  = tfrdat[diff2,:]
    diff4dat  = tfrdat[diff4,:]
    diff8dat  = tfrdat[diff8,:]
    diff12dat = tfrdat[diff12,:]
    
    gmean[count]        = tfrdat.copy().mean(axis=0)
    correct[count]      = corrdat.mean(axis=0)
    incorrect[count]    = incorrdat.mean(axis=0)
    
    d2[count]   = diff2dat.mean(axis=0)
    d4[count]   = diff4dat.mean(axis=0)
    d8[count]   = diff8dat.mean(axis=0)
    d12[count]  = diff12dat.mean(axis=0)
#%%

#plot some things here

fig = plt.figure()
ax = fig.add_subplot(111)
ax.axvspan(xmin=-0.0001, xmax = 0, color = '#000000', alpha = 0.9)
ax.plot(times, gmean.mean(axis=0), label = 'grandmean', color = '#3182bd')
ax.fill_between(times,
                y1 = np.subtract(gmean.mean(axis=0), sp.stats.sem(gmean, axis =0, ddof = 0)),
                y2 = np.add(gmean.mean(axis=0), sp.stats.sem(gmean, axis =0, ddof = 0)), color = '#3182bd', alpha = 0.3)
fig.legend()


#plot by subsequent correctness

fig = plt.figure()
ax = fig.add_subplot(111)
ax.axvspan(xmin=-0.0001, xmax = 0, color = '#000000', alpha = 0.9)

ax.plot(times, correct.mean(axis=0), label = 'correct', color = '#31a354')
ax.fill_between(times,
                y1 = np.subtract(correct.mean(axis=0), sp.stats.sem(correct, axis =0, ddof = 0)),
                y2 = np.add(correct.mean(axis=0), sp.stats.sem(correct, axis =0, ddof = 0)), color = '#31a354', alpha = 0.3)
ax.plot(times, incorrect.mean(axis=0), label = 'incorrect', color = '#f03b20')
ax.fill_between(times,
                y1 = np.subtract(incorrect.mean(axis=0), sp.stats.sem(incorrect, axis =0, ddof = 0)),
                y2 = np.add(incorrect.mean(axis=0), sp.stats.sem(incorrect, axis =0, ddof = 0)), color = '#f03b20', alpha = 0.3)
fig.legend()


#plot by difficulty

fig = plt.figure()
ax = fig.add_subplot(111)
ax.axvspan(xmin=-0.0001, xmax = 0, color = '#000000', alpha = 0.9)
ax.plot(times, d2.mean(axis=0), label = 'difficulty 2', color = '#d7191c')
ax.fill_between(times,
                y1 = np.subtract(d2.mean(axis=0), sp.stats.sem(d2, axis =0, ddof = 0)),
                y2 = np.add(d2.mean(axis=0), sp.stats.sem(d2, axis =0, ddof = 0)), color = '#d7191c', alpha = 0.3)
ax.plot(times, d4.mean(axis=0), label = 'difficulty 4', color = '#fdae61')
ax.fill_between(times,
                y1 = np.subtract(d4.mean(axis=0), sp.stats.sem(d4, axis =0, ddof = 0)),
                y2 = np.add(d4.mean(axis=0), sp.stats.sem(d4, axis =0, ddof = 0)), color = '#fdae61', alpha = 0.3)
ax.plot(times, d8.mean(axis=0), label = 'difficulty 8', color = '#a6d96a')
ax.fill_between(times,
                y1 = np.subtract(d8.mean(axis=0), sp.stats.sem(d8, axis =0, ddof = 0)),
                y2 = np.add(d8.mean(axis=0), sp.stats.sem(d8, axis =0, ddof = 0)), color = '#a6d96a', alpha = 0.3)
ax.plot(times, d12.mean(axis=0), label = 'difficulty 12', color = '#1a9641')
ax.fill_between(times,
                y1 = np.subtract(d12.mean(axis=0), sp.stats.sem(d12, axis =0, ddof = 0)),
                y2 = np.add(d12.mean(axis=0), sp.stats.sem(d12, axis =0, ddof = 0)), color = '#1a9641', alpha = 0.3)
fig.legend()

