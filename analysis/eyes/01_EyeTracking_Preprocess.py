#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:16:55 2024

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
np.set_printoptions(suppress=True)

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo
from eyefuncs_mne import find_blinks, transform_pupil

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

eyedir  = op.join(wd, 'data', 'eyes', 'asc')
datadir = op.join(wd, 'data', 'datafiles', 'combined') #path to single subject datafiles
figdir  = op.join(wd, 'figures', 'eyes') 
trlcheckdir = op.join(wd, 'data', 'eyes', 'trialchecks', 'stim1locked')

drop_gaze = True

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
# subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35,             39])
# subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now

icount = -1
for i in subs:
    icount += 1
    print(f"\n- - - - working on participant {i} - - - -")
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    
    if not op.exists(param['s1locked_eyes']):
        #epoch the data
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
        event_id = { 'trig1':   1, 'trig10': 10,        #stim1
                     'trig2':   2,  'trig3':  3,       #stim2
                     'trig11': 11, 'trig12': 12, 
                     'trig22': 22, 'trig23': 23,  #response cue
                     'trig31': 31, 'trig32': 32,  
                     'trig40': 40, #button pressleft
                     'trig50': 50, #button press right
                      'trig60':  60, #correct feedback
                      'trig61':  61, #incorrect feedback
                      'trig62':  62, #timed out feedback,
                     'trig150': 150, #ITI trigger
                     'trigPLR_ON': 200,
                     
                     'blink':      201,
                     'saccade':    202,
                     'fixation':   203
                     }
        
        if i not in [29,30]:
            raw = mne.io.read_raw_fif(fname = param['eyes_preproc'], preload=True) #get raw data with blinks identified
            bdata = pd.read_csv(param['behaviour'], index_col=False) #get behavioural data for participant
            
            #crop out the PLR section as this will massively bias estimate of on-task mean pupil size
            #we want to crop out the PLR phase if possible
            plrannots = [x for x in raw.annotations if 'PLR' in x['description']]
            last_plr = plrannots[-1]['onset']
            first_trl = [x for x in raw.annotations if x['description'] == 'trig150'][0]['onset']
            tmin_crop = first_trl - 5 #get 5s before the first trial of the task
            raw.crop(tmin = tmin_crop) #crops out all the pre-task stuff so zscoring later isnt affected by the PLR measurement
            
            raw = mne.preprocessing.eyetracking.interpolate_blinks(raw, buffer = 0.0) #interpolate over blinks.
            # raw.plot(scalings=dict(eyegaze=1e3, pupil = 50e1))
            raw = transform_pupil(raw) #express pupil as a % of mean pupil size
            
            # if you filter before interpolating, then the filter will spread nans across the entire filter window which messes data up even more
            # low pass filtering to remove high frequency tremor should only happen after interpolation to avoid this
            #raw = raw.filter(l_freq = None, h_freq = 50, picks = ['pupil', 'pupil_transformed']) #lowpass at 50Hz to remove some tremor in the signal
            
            events, _ = mne.events_from_annotations(raw, event_id)
        
            stim1locked = mne.Epochs(raw, events, events_s1, tmin, tmax, baseline, reject_by_annotation=False, preload = True)
            if i == 37:
                bdata = bdata.query('blocknumber in [1,2,3]') #participant withdrew after 3 blocks of the task
            
            
            stim1locked.metadata = bdata
            stim1locked.save(fname = param['s1locked_eyes'], overwrite = True)
        
        if i in [29, 30]:
            
            raw = mne.io.read_raw_fif(fname = param['eyes_preproc'], preload=True) #get raw data with blinks identified
            raw2 = mne.io.read_raw_fif(fname = param['eyes_preproc'].replace('a_preproc', 'b_preproc'), preload = True)
            bdata = pd.read_csv(param['behaviour'], index_col=False)
    
            # remove PLR segments from the first recording 
            plrannots = [x for x in raw.annotations if 'PLR' in x['description']]
            last_plr = plrannots[-1]['onset']
            first_trl = [x for x in raw.annotations if x['description'] == 'trig150'][0]['onset']
            tmin_crop = first_trl - 5 #get 5s before the first trial of the task
            raw.crop(tmin = tmin_crop)
            
            #remove PLR segments from the second recording too, if present
            plrannots = [x for x in raw2.annotations if 'PLR' in x['description']]
            if len(plrannots) > 0:
                last_plr = plrannots[-1]['onset']
                first_trl = [x for x in raw2.annotations if x['description'] == 'trig150'][0]['onset']
                tmin_crop = first_trl - 5 #get 5s before the first trial of the task
                raw2.crop(tmin = tmin_crop)
            
            raw = mne.preprocessing.eyetracking.interpolate_blinks(raw, buffer = 0.0) #interpolate over blinks.
            raw2 = mne.preprocessing.eyetracking.interpolate_blinks(raw2, buffer = 0.0) #interpolate over blinks.

            
            
            raw = transform_pupil(raw)
            raw2 = transform_pupil(raw2)
                
            events,  _ = mne.events_from_annotations(raw, event_id)
            events2, _ = mne.events_from_annotations(raw2,event_id)
            
            stim1locked  = mne.Epochs(raw,  events,  events_s1,   tmin, tmax, baseline, reject_by_annotation=False, preload=True)
            stim1locked2 = mne.Epochs(raw2, events2, events_s1, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
            
            if i == 29:
                bdata1                = bdata.copy()[:len(stim1locked)]
                stim1locked.metadata  = bdata1
                stim1locked           = stim1locked['blocknumber in [1,2]']
                bdata2                = bdata.copy().query('blocknumber in [3,4,5]')
                stim1locked2.metadata = bdata2
                
                stim1locked = mne.concatenate_epochs([stim1locked, stim1locked2])
            if i == 30:
                stim1locked.metadata  = bdata.copy().query('blocknumber in [1,2,3]')
                stim1locked2.metadata = bdata.copy().query('blocknumber in [4,5]')
                stim1locked = mne.concatenate_epochs([stim1locked, stim1locked2])
        
        stim1locked.metadata = bdata
        stim1locked.save(fname = param['s1locked_eyes'], overwrite = True)