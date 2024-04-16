#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:32:48 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import sklearn as skl
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
%matplotlib
mne.viz.set_browser_backend('qt')
import glmtools as glm

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gesd, plot_AR

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
#%%

for i in subs:
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    print(f'- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp'), preload=True)
    epochs = epochs.apply_baseline((-0.2, 0)) #baseline just prior to stim1 onset
    
    
    
    #run gesd
    #do trial rejection from the two files separately before concatenating events
    _, keeps = plot_AR(deepcopy(epochs).pick_types(eeg=True),
                       method = 'gesd',
                       zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
    keeps = keeps.flatten() #indices of trials to be kept

    discards = np.ones(len(epochs), dtype = 'bool')
    discards[keeps] = False #vector that is False if kept, True if discarded
    epochs = epochs.drop(discards) #first we'll drop trials with excessive noise in the EEG
    
    #write to file which trials are discarded from the eeg
    
    #go through any remaining trials to look for excessively noisy trials to discard
    # epoched.plot(events = epoched.events, n_epochs = 3, n_channels = 64, scalings = dict(eeg=40e-6))
    #epoched.interpolate_bads()
    
    #save the epoched data, combined with metadata, to file
    epochs.save(fname = param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'), overwrite=True)
    #save the resulting behavioural data too
    
    
    del(epochs)
    plt.close('all')