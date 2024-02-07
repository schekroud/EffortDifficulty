# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

fig_allsubs = plt.figure(figsize = [15,15])
axcount = 1
for i in subs:
    print(f"\n- - - - working on participant {i} - - - -")
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    
    # raw = mne.io.read_raw_eyelink(fname = param['asc'],
    #                               create_annotations = True) #creates blink, saccade, fixation events + stim triggers
    raw = mne.io.read_raw_eyelink(fname = param['asc'], create_annotations = ['fixations', 'saccades', 'messages'])
    raw.drop_channels(['DIN'])
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

    if i in [29, 30]:
        raw2 = mne.io.read_raw_eyelink(fname = param['asc'].replace('a.asc', 'b.asc'), create_annotations = ['fixations', 'saccades', 'messages'])
        raw2.drop_channels(['DIN'])
        xpos_chan  = [x for x in raw2.ch_names if 'xpos' in x][0]
        ypos_chan  = [x for x in raw2.ch_names if 'ypos' in x][0]
        pupil_chan = [x for x in raw2.ch_names if 'pupil' in x][0]
        
        raw2.rename_channels({
            xpos_chan:'xpos',
            ypos_chan:'ypos',
            pupil_chan:'pupil'
            })
    '''
    when you use the automated routines in mne python, they use the SR Research event detections for saccades, fixations and blinks
    for blinks, it works on some, but misses lots
    specifically, there are instances of 'half blinks' where the eyes dont fully close, so the trace never fully goes to nan
    this means its not detected as a blink, but instead detected as a rapid change in the trace - a saccade.
    when you actually inspect the traces you see that this isnt plausible - the effect of a saccade on the pupil is substantially smaller
    
    so instead, run personal routine on the data to identify blinks agnostic to the blink/saccade detection of the eyetracker    
    '''
    raw = find_blinks(raw, add_nanchannel=True)
    if i in [29, 30]:
        raw2 = find_blinks(raw2, add_nanchannel=True)
    # raw.plot(scalings = dict(eyegaze=1e3))
    
    #epoch the data
    events_s1       = [1, 10] #stim1 triggers
    
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
    events, _ = mne.events_from_annotations(raw, event_id)
    
    stim1locked = mne.Epochs(raw, events, events_s1, tmin, tmax, baseline, reject_by_annotation=False, preload = True)
    
    if i in [29, 30]:
        events2, _ = mne.events_from_annotations(raw2, event_id)
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

    ax=fig_allsubs.add_subplot(6,5,axcount)
    ax.set_title(f"subject {i}")
    stim1locked.plot_image(picks='pupil_nan', evoked=False, colorbar=False, cmap='viridis', axes=ax, title = f"subject {i}")
    axcount +=1
    
    
    #for each subject, we want to get the % of data that is nan in a given trial to help decide whether to exclude a trial
    s1dat = np.squeeze(stim1locked.get_data(picks='pupil_nan'))
    trlnans = np.isnan(s1dat).sum(axis=1) #number of nan data points per trial
    trlnans_perc = np.round(np.divide(trlnans, stim1locked.times.size)*100, 2) #round to 2dp for *reasons*
    np.save(file = op.join(trlcheckdir, f"{param['subid']}_stim1locked_eyes_nancheck_perTrial.npy"), arr = trlnans_perc)
    
    # save the new raw object to file - it contains the slightly better blink detection
    # and has a channel where the data is set to nan for these blinks already
    raw.save(fname = param['eyes_preproc'], fmt = 'double', overwrite = True)
    if i in [29,30]:
        raw2.save(fname = param['eyes_preproc'].replace('a_preproc', 'b_preproc'), fmt = 'double', overwrite = True)
    
    
fig_allsubs.tight_layout()
fig_allsubs.savefig(fname = op.join(figdir, 'stim1locked_allTrials_showMissing.pdf'), format = 'pdf', dpi = 300)

#looking at this, probably want to exclude participants 19, maybe 20, query 13/14/31/34 ?
#%%
