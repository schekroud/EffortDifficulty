#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:27:16 2023

@author: sammi
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

# sys.path.insert(0, '/ohba/pi/knobre/schekroud/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gesd, plot_AR

# wd = '/ohba/pi/knobre/schekroud/postdoc/student_projects/EffortDifficulty' #workstation wd
wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

subs = np.array([10, 11])

for i in subs:
    print('\nworking on subject ' + str(i) +'\n')
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    
    raw = mne.io.read_raw_fif(fname = param['eeg_preproc'], preload = False)
    raw.set_montage('easycap-M1', on_missing='raise', match_case=False) #just make sure montage is applied
    
    events_s1       = [1, 10] #stim1 triggers
    events_s2       = [2, 3, 11, 12] #stim2 triggers
    events_respc    = [22, 23, 31, 32] #response cue triggers
    events_button   = [40, 50] #button-press triggers
    events_feedback = [60, 61, 62]
    events_iti      = [150]
    events_plr      = [200]
    
    #epoching to stim1 onset
    
    tmin, tmax = -2, 2
    baseline = None #don't baseline just yet
    
    
    #get events from annotations
    event_id = { '1':   1, '10': 10,        #stim1

                 '2':   2,  '3':  3,       #stim2
                 '11': 11, '12': 12, 
                 
                 '22': 22, '23': 23,  #response cue
                 '31': 31, '32': 32,  
                 
                 '40': 40, #button pressleft
                 '50': 50, #button press right
                 
                  '60':  60, #correct feedback
                  '61':  61, #incorrect feedback
                  '62':  62, #timed out feedback,
                 '150': 150, #ITI trigger
                 '200': 200 #PLR trigger
                 }
    events, _ = mne.events_from_annotations(raw, event_id)
    epoched = mne.Epochs(raw, events, events_s2, tmin, tmax, baseline, reject_by_annotation=False, preload=False)
    bdata = pd.read_csv(param['behaviour'], index_col = None)
    epoched.metadata = bdata
    epoched.save(fname = param['stim2locked'], overwrite = True)
    
    #remove from RAM
    del(epoched)
    del(raw)