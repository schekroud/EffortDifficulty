#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:26:34 2023

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
%matplotlib

# sys.path.insert(0, '/ohba/pi/knobre/schekroud/postdoc/student_projects/perceptdiff/analysis_scripts/tools')
sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gesd, plot_AR

wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)


subs = np.array([10, 11])
for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    
    #read in the already ica cleaned + epoched data
    epoched = mne.read_epochs(fname = param['stim2locked'], preload = True)

    #do trial rejection from the two files separately before concatenating events
    _, keeps = plot_AR(deepcopy(epoched).pick_types(eeg=True),
                       method = 'gesd',
                       zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
    keeps = keeps.flatten() #indices of trials to be kept

    discards = np.ones(len(epoched), dtype = 'bool')
    discards[keeps] = False #vector that is False if kept, True if discarded
    epoched = epoched.drop(discards) #first we'll drop trials with excessive noise in the EEG
    
    #write to file which trials are discarded from the eeg
    
    #go through any remaining trials to look for excessively noisy trials to discard
    epoched.plot(events = epoched.events, n_epochs = 3, n_channels = 64, scalings = dict(eeg=40e-6))
    #epoched.interpolate_bads()
    
    #save the epoched data, combined with metadata, to file
    epoched.save(fname = param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned'), overwrite=True)
    #save the resulting behavioural data too
    epoched.metadata.to_csv(param['behaviour'].replace('combined', 'stim2locked_combined_eegcleaned'))
    
    del(epoched)
    plt.close('all')