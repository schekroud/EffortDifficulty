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
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gesd, plot_AR

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
# subs = np.array([                                                                                                    35, 36, 37, 38, 39])
for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    
    events_s1       = [1, 10] #stim1 triggers
    events_s2       = [2, 3, 11, 12] #stim2 triggers
    events_respc    = [22, 23, 31, 32] #response cue triggers
    events_button   = [40, 50] #button-press triggers
    events_feedback = [60, 61, 62]
    events_iti      = [150]
    events_plr      = [200]
    
    #epoching to stim1 onset
    
    tmin, tmax = -3, 2
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
    
    if i not in [29, 30]:
        raw = mne.io.read_raw_fif(fname = param['eeg_preproc'], preload = False)
        raw.set_montage('easycap-M1', on_missing='raise', match_case=False) #just make sure montage is applied
        events, _ = mne.events_from_annotations(raw, event_id)
        
        if i == 12:
            #one trigger is labelled wrong 
            wrongid = np.where(events[:,0] ==1325388)[0] #it received a 10 when 11 was sent
            events[wrongid,2] = 11 #re-label this trigger
            
        epoched = mne.Epochs(raw, events, events_s2, tmin, tmax, baseline, reject_by_annotation=False, preload=False)
        bdata = pd.read_csv(param['behaviour'], index_col = None)
        if i == 37:
            bdata = bdata.query('blocknumber in [1,2,3]') #participant withdrew after 3 blocks of task
        
        epoched.metadata = bdata
        epoched.save(fname = param['stim2locked'], overwrite = True)
        
        #remove from RAM
        del(epoched)
        del(raw)
    elif i == 29:
        raw = mne.io.read_raw_fif(fname = param['eeg_preproc'], preload = False)
        raw.set_montage('easycap-M1', on_missing='raise', match_case=False) #just make sure montage is applied
        
        events, _ = mne.events_from_annotations(raw, event_id)
        epoched = mne.Epochs(raw, events, events_s2, tmin, tmax, baseline,
                             reject_by_annotation=False, preload=False, on_missing='ignore') #it doesn't like to continue if it doesn't find a trigger
        
        raw2 = mne.io.read_raw_fif(fname = param['eeg_preproc'].replace('_preproc-raw', 'b_preproc-raw'), preload = False)
        raw2.set_montage('easycap-M1', on_missing='raise', match_case=False) #just make sure montage is applied
        
        events2, _ = mne.events_from_annotations(raw2, event_id)
        epoched2 = mne.Epochs(raw2, events2, events_s2, tmin, tmax, baseline,
                             reject_by_annotation=False, preload=False, on_missing='ignore') #it doesn't like to continue if it doesn't find a trigger
        #need to cut down 
        bdata = pd.read_csv(param['behaviour'], index_col=False)
        
        bdata1 = bdata.copy()[:len(epoched.events)] #only blocks 1/2 from this file truly apply to this recording
        epoched.metadata = bdata1 #assign the behavioural data into the eeg data
        epoched = epoched['blocknumber in [1,2]'] #because we only have 2 blocks of usable data, just take the two blocks
        
        bdata2 = bdata.copy().query('blocknumber in [3,4,5]')
        epoched2.metadata = bdata2
        
        epochs = mne.concatenate_epochs([epoched, epoched2]) #combine into one file, finally
        epochs.metadata = bdata #finally just ensure the metadata is correct
        epochs.save(fname = param['stim2locked'], overwrite = True)
        
        #remove from RAM
        del(epochs)
        del(raw)
        del(raw2)
        
        
    elif i == 30:
        raw = mne.io.read_raw_fif(fname = param['eeg_preproc'], preload = False)
        raw.set_montage('easycap-M1', on_missing='raise', match_case=False) #just make sure montage is applied
        
        events, _ = mne.events_from_annotations(raw, event_id)
        epoched = mne.Epochs(raw, events, events_s2, tmin, tmax, baseline,
                             reject_by_annotation=False, preload=False, on_missing='ignore') #it doesn't like to continue if it doesn't find a trigger
        
        raw2 = mne.io.read_raw_fif(fname = param['eeg_preproc'].replace('_preproc-raw', 'b_preproc-raw'), preload = False)
        raw2.set_montage('easycap-M1', on_missing='raise', match_case=False) #just make sure montage is applied
        
        events2, _ = mne.events_from_annotations(raw2, event_id)
        epoched2 = mne.Epochs(raw2, events2, events_s2, tmin, tmax, baseline,
                             reject_by_annotation=False, preload=False, on_missing='ignore') #it doesn't like to continue if it doesn't find a trigger
        
        bdata = pd.read_csv(param['behaviour'])
        
        bdata1 = bdata.query('blocknumber in [1,2,3]')
        #this dataset stopped after 3 blocks, so we have a first recording with 3 complete blocks, and a second recording with 2 complete blocks
        epoched.metadata = bdata1 #assign the behavioural data into the eeg data
       
        #the second recording has two full blocks of data in, so we'll read in those datafiles and add them in
        bdata2 = bdata.query('blocknumber in [4,5]')
        epoched2.metadata = bdata2
                
        epochs = mne.concatenate_epochs([epoched, epoched2]) #combine into one file, finally
        epochs.save(fname = param['stim2locked'], overwrite = True)
        
        #remove from RAM
        del(epochs)
        del(raw)
        del(raw2)