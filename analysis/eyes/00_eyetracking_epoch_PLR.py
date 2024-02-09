#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:00:23 2024

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

drop_gaze = True

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35, 36, 37, 38, 39]) #29, 30 need some fixing
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35,         38, 39]) #36/37 unusable
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35,             39])
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now

for i in subs:
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    
    # raw = mne.io.read_raw_eyelink(fname = param['asc'],
    #                               create_annotations = True) #creates blink, saccade, fixation events + stim triggers
    raw = mne.io.read_raw_eyelink(fname = param['asc'], create_annotations = ['fixations', 'saccades', 'messages'])
    bdata = pd.read_csv(op.join(param['path'], 'datafiles','combined', 'EffortDifficulty_%s_combined_py.csv'%param['subid']))
    
    #some small steps that will just make things a bit easier in future (harmonising individuals where the left or right were recorded, which can vary)
    xpos_chan  = [x for x in raw.ch_names if 'xpos' in x][0]
    ypos_chan  = [x for x in raw.ch_names if 'ypos' in x][0]
    pupil_chan = [x for x in raw.ch_names if 'pupil' in x][0]
    
    raw.rename_channels({
        xpos_chan:'xpos',
        ypos_chan:'ypos',
        pupil_chan:'pupil'
        })
    raw.drop_channels(['DIN', 'xpos', 'ypos'])

    
    '''
    when you use the automated routines in mne python, they use the SR Research event detections for saccades, fixations and blinks
    for blinks, it works on some, but misses lots
    specifically, there are instances of 'half blinks' where the eyes dont fully close, so the trace never fully goes to nan
    this means its not detected as a blink, but instead detected as a rapid change in the trace - a saccade.
    when you actually inspect the traces you see that this isnt plausible - the effect of a saccade on the pupil is substantially smaller
    
    so instead, run personal routine on the data to identify blinks agnostic to the blink/saccade detection of the eyetracker    
    '''

    raw = find_blinks(raw)
    raw = mne.preprocessing.eyetracking.interpolate_blinks(raw, buffer = 0.0) #interpolate over blinks.
    # raw.plot(scalings = dict(eyegaze=1e3))
    
    #epoch the data
    events_s1       = [1, 10] #stim1 triggers
    events_s2       = [2, 3, 11, 12] #stim2 triggers
    events_respc    = [22, 23, 31, 32] #response cue triggers
    events_button   = [40, 50] #button-press triggers
    events_feedback = [60, 61, 62]
    events_iti      = [150]
    events_plr      = [200]
    
    #epoching to stim1 onset
    tmin, tmax = -2, 5
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
    events, _ = mne.events_from_annotations(raw, event_id)
    
    epoched = mne.Epochs(raw, events, events_plr, tmin, tmax, baseline, reject_by_annotation=False, preload = True)
    epoched.drop_channels(['pupil_nan'])
    
    epoched.save(param['plrlocked'], overwrite = True)
    
    ave = np.squeeze(epoched._data.mean(axis=0)) #average across PLRS
    times = epoched.times
    
    fig = plt.figure(figsize = [4,3])
    ax = fig.add_subplot(111)
    ax.plot(times, ave)
    ax.set_title(f"participant {i}")
    
    
    
    