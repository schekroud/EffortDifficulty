#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:58:33 2024

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
%matplotlib
mne.viz.set_browser_backend('qt')

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo
from eyefuncs_mne import find_blinks

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

eyedir  = op.join(wd, 'data', 'eyes', 'asc')
datadir = op.join(wd, 'data', 'datafiles', 'combined') #path to single subject datafiles
figdir  = op.join(wd, 'figures', 'eyes') 
trlcheckdir = op.join(wd, 'data', 'eyes', 'trialchecks', 'stim1locked')

drop_gaze = True

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
# subs = [10]
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now

chan2use = 'pupil_nan'
bwin = [-1, -0.5]

for i in subs:
    print(f"\n- - - - working on participant {i} - - - -")
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    
    epoched = mne.epochs.read_epochs(param['s1locked_eyes'], preload = True)
    nonchans = [x for x in epoched.ch_names if x != chan2use]
    
    epoched.drop_channels(nonchans) 
    
    epoched = epoched.crop(tmin = bwin[0], tmax = bwin[1])
    eyedat  = np.squeeze(epoched._data)
    eyedat  = np.nanmean(eyedat, axis = 1) #average across time to get single baseline value for this trial
    
    
    np.save(file = op.join(eyedir, '%s_eyes_stim1locked_baselinevalues.npy'%(param['subid'])), arr = eyedat)
