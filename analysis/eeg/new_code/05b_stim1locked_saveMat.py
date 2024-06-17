#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:17:02 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
%matplotlib

loc = 'laptop'
if loc == 'laptop':
    sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
elif loc == 'workstation':
    sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

from funcs import getSubjectInfo
os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
for i in subs:
    sub   = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    print(f'\n- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'),
                                verbose = 'ERROR', preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -0.5, tmax = 1.5) #crop for speed and relevance  
    # epochs.resample(100)
    # times = epochs.times
    
    if i == 16:
        epochs = epochs['blocknumber != 4']
    if i in [21, 25]:
        #these participants had problems in block 1
        epochs = epochs['blocknumber > 1']
    
    #drop all but the posterior visual channels
    epochs = epochs.pick(picks = [
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                        'O1', 'Oz', 'O2'])
    data   = epochs._data.copy() #get the data matrix
    bdata  = epochs.metadata.copy() 
    
    #get vectors to save alongside the data matrix
    orientations = bdata.stim1ori.to_numpy()
    trlid        = bdata.trlid.to_numpy()
    difficulty   = bdata.difficultyOri.to_numpy()
    
    matdict = dict()
    matdict['data']  = data
    matdict['oris']  = orientations
    matdict['trlid'] = trlid
    matdict['diff']  = difficulty
    
    #save data to matlab file
    sp.io.savemat(file_name = op.join(wd, 'data', 'eeg', 'mat', f'EffortDifficulty_s{i}_stim1locked.mat'),
                  mdict = matdict, appendmat = False)
    
    