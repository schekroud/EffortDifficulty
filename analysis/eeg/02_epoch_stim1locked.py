#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:41:03 2023

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

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')

from funcs import getSubjectInfo, gesd, plot_AR

def streaks_numbers(array):
    '''
    finds streaks of an array where a number is the same
    - useful for finding sequences of trials where difficulty is the same
    note: bit slow because it literally loops over an array so it isn't the fastest, but it works
    '''
    x = np.zeros(array.size).astype(int)
    count = 0
    for ind in range(len(x)):
        if array[ind] == array[ind-1]:
            count += 1 #continuing the sequence
        else: #changed difficulty
            count = 1
        x[ind] = count
    return x

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    
    raw = mne.io.read_raw_fif(fname = param['eeg_preproc'], preload = False)
    # tmp = mne.io.read_raw_curry(fname = param['raweeg'], preload = False)
    raw.set_montage('easycap-M1', on_missing='raise', match_case=False) #just make sure montage is applied
    
    events_s1       = [1, 10] #stim1 triggers
    events_s2       = [2, 3, 11, 12] #stim2 triggers
    events_respc    = [22, 23, 31, 32] #response cue triggers
    events_button   = [40, 50] #button-press triggers
    events_feedback = [60, 61, 62]
    events_iti      = [150]
    events_plr      = [200]
    
    #epoching to stim1 onset
    
    tmin, tmax = -3, 4
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
                 }
    events, _ = mne.events_from_annotations(raw, event_id)
    
    if i == 12:
        #one trigger is labelled wrong 
        wrongid = np.where(events[:,0] ==1325388)[0] #it received a 10 when 11 was sent
        events[wrongid,2] = 11 #re-label this trigger
        
    epoched = mne.Epochs(raw, events, events_s1, tmin, tmax, baseline, reject_by_annotation=False, preload=False)
    bdata = pd.read_csv(param['behaviour'], index_col = None)
    
    bdat = pd.DataFrame()
    for iblock in bdata.blocknumber.unique():
        #this needs to be run separately for each block 
        tmpdata = bdata.query('blocknumber == @iblock')
        #add in where this trial is within a perceptual difficulty sequence
        tmpdata = tmpdata.assign(diffseqpos = streaks_numbers(tmpdata.difficultyOri.to_numpy()))
        #reconstruct the behavioural data file
        bdat = pd.concat([bdat, tmpdata])
        # bdata = bdata.assign(diffseqpos = streaks_numbers(bdata.difficultyOri.to_numpy()))
    
    epoched.metadata = bdat
    epoched.save(fname = param['stim1locked'], overwrite = True)
    
    #remove from RAM
    del(epoched)
    del(raw)