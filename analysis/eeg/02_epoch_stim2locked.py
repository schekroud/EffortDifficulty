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

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
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
                 '200': 200 #PLR trigger
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
        
        datdir = op.join(wd, 'data', 'datafiles', 's29')
        filelist = os.listdir(datdir)
        flist1 = [x for x in filelist if 's29b' not in x]
        flist2 = [x for x in filelist if 's29b'     in x]
    
        bdata1 = pd.DataFrame()
        for file in flist1:
            tmpdf = pd.read_csv(op.join(datdir, file), index_col=False)
            bdata1 = pd.concat([bdata1, tmpdf])
        #this dataset crashed shortly after the start of the third block, so we have 3 blocks of behavioural data (most of which is just missing data)
        #cut it down to just the trials with data
        bdata1 = bdata1[:len(epoched.events)]
        epoched.metadata = bdata1 #assign the behavioural data into the eeg data
        epoched = epoched['blocknumber in [1,2]'] #because we only have 2 blocks of usable data, just take the two blocks
        
    
        #the second recording has three full blocks of data in, so we'll read in those datafiles and add them in
        bdata2 = pd.DataFrame()
        for file in flist2:
            tmpdf = pd.read_csv(op.join(datdir, file), index_col=False)
            bdata2 = pd.concat([bdata2, tmpdf])
        epoched2.metadata = bdata2
        
        epochs = mne.concatenate_epochs([epoched, epoched2]) #combine into one file, finally
        bdata = epochs.metadata.copy()
        bdata = bdata.assign(trlid = np.add(np.arange(len(bdata)),1)) #assign proper trial numbers
        blocks=np.arange(1,6,1)
        bdata = bdata.assign(blocknumber = np.sort(np.tile(blocks, int(len(bdata)/blocks.size)))) #add proper block number in        
        bdat = pd.DataFrame()
        for iblock in bdata.blocknumber.unique():
            #this needs to be run separately for each block 
            tmpdata = bdata.query('blocknumber == @iblock')
            #add in where this trial is within a perceptual difficulty sequence
            tmpdata = tmpdata.assign(diffseqpos = streaks_numbers(tmpdata.difficultyOri.to_numpy()))
            #reconstruct the behavioural data file
            bdat = pd.concat([bdat, tmpdata])
            # bdata = bdata.assign(diffseqpos = streaks_numbers(bdata.difficultyOri.to_numpy()))
        
        #need to add in a few other things too
        bdat = bdat.assign(subid = i)
        bdat = bdat.assign(fbgiven = 'timed out')
        bdat['fbgiven'] = np.select([np.isin(bdat.fbtrig, [60]), np.isin(bdat.fbtrig, [61])],
                                     ['correct',                 'incorrect'],
                                     default = 'timed out')        
        bdat = bdat.assign(prevtrlfb = bdat.fbgiven.shift(1)) #get previous trial feedback
        bdat = bdat.assign(rewarded = np.where(bdat.fbgiven == 'correct', 1, 0))
        bdat = bdat.assign(unrewarded = np.where(bdat.fbgiven == 'incorrect', 1, 0))    
        epochs.metadata = bdat
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
        bdata = epochs.metadata.copy()
        bdat = pd.DataFrame()
        for iblock in bdata.blocknumber.unique():
            #this needs to be run separately for each block 
            tmpdata = bdata.query('blocknumber == @iblock')
            #add in where this trial is within a perceptual difficulty sequence
            tmpdata = tmpdata.assign(diffseqpos = streaks_numbers(tmpdata.difficultyOri.to_numpy()))
            #reconstruct the behavioural data file
            bdat = pd.concat([bdat, tmpdata])
            # bdata = bdata.assign(diffseqpos = streaks_numbers(bdata.difficultyOri.to_numpy()))
    
        epochs.metadata = bdat
        epochs.save(fname = param['stim2locked'], overwrite = True)
        
        #remove from RAM
        del(epochs)
        del(raw)
        del(raw2)