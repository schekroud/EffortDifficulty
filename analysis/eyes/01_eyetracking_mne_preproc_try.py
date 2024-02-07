#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:00:07 2024

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
gave        = np.zeros(shape = [subs.size, 7001])
correct     = np.zeros_like(gave) #7001 timepoints for -3s -> 4s at 1KHz
incorrect   = np.zeros_like(gave)
diff        = np.zeros_like(gave)

thetas = [2,4,8,12]
difficulties = np.zeros(shape = [subs.size, len(thetas), 7001])
diff2 = np.zeros(shape = (subs.size, 2, 7001))
diff4 = np.zeros_like(diff2)
diff8 = np.zeros_like(diff2)
diff12 = np.zeros_like(diff2)

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
    else:
        stim1locked = mne.epochs.read_epochs(param['s1locked_eyes'], preload=True)
    times = stim1locked.times
        
    if i ==16:
        stim1locked = stim1locked['blocknumber != 4'] #lost the eye in block four of this participant
    if i in [21,25]:
        stim1locked = stim1locked['blocknumber > 1'] #these ppts had problems in block 1 of the task
    
    if i == 26:
        stim1locked = stim1locked['trlid != 192'] #this trial has large amounts of interpolated data, and some nans persist
    stim1locked = stim1locked['fbtrig != 62'] #drop timeout trials
    
    baseline = None
    # baseline = [-0.2, 0]
    
    
    # s1corr = stim1locked['rewarded == 1'].pick(picks='pupil_transformed').apply_baseline(baseline).average()
    # s1incorr = stim1locked['unrewarded == 1'].pick(picks='pupil_transformed').apply_baseline(baseline).average()
    
    # corr.append(s1corr)
    # incorr.append(s1incorr)
    
    tmpgave = np.squeeze(stim1locked.pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
    tmpgave_mean = np.nanmean(tmpgave, axis=0)
    gave[icount] = tmpgave_mean
    
    # fig = plt.figure(figsize = [4,3])
    # ax = fig.add_subplot(111)
    # ax.plot(times, tmpgave.T)
    # ax.set_title(f"participant {i}")
    
    
    tmpcorr = np.squeeze(stim1locked['rewarded==1'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
    tmpcorr = np.nanmean(tmpcorr, axis=0) #average across trials
    
    tmpincorr = np.squeeze(stim1locked['unrewarded==1'].pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
    tmpincorr = np.nanmean(tmpincorr, axis=0)
    
    tmpdiff = np.subtract(tmpcorr, tmpincorr)
    
    correct[icount]   = tmpcorr
    incorrect[icount] = tmpincorr
    diff[icount]      = tmpdiff
    
    diffcount = -1
    for theta in thetas: #loop over orientation difficulties
        diffcount +=1
        inds = np.where(stim1locked.metadata.difficultyOri==theta)[0]
        tmp = stim1locked.copy()[inds] #get just trials of this difficulty
        tmpdata = np.squeeze(tmp.pick(picks='pupil_transformed').apply_baseline(baseline).get_data())
        tmpdata = np.nanmean(tmpdata, axis=0)
        difficulties[icount, diffcount] = tmpdata
        
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(stim1locked.times, np.squeeze(s1corr.data),   label = 'correct',   color = '#2ca25f')
    # ax.plot(stim1locked.times, np.squeeze(s1incorr.data), label = 'incorrect', color = '#e34a33')
    # ax.legend(loc = 'lower left')
    # ax.axvline(0, lw = 1, color = '#000000', ls = 'dashed')
    # ax.axhline(0, lw = 1, color = '#000000', ls = 'dashed')
#%%

gave_mean, gave_sem = np.nanmean(gave, axis=0), sp.stats.sem(gave, axis=0, ddof= 0, nan_policy='omit')

corr_mean,     corr_sem = np.nanmean(correct, axis=0)  , sp.stats.sem(correct, axis=0, ddof=0, nan_policy='omit')
incorr_mean, incorr_sem = np.nanmean(incorrect, axis=0), sp.stats.sem(incorrect, axis=0, ddof=0, nan_policy='omit')
diff_mean, diff_sem     = np.nanmean(diff, axis=0), sp.stats.sem(diff, axis=0, ddof=0, nan_policy='omit')

difficulties_mean, difficulties_sem = np.nanmean(difficulties, axis=0), sp.stats.sem(difficulties, axis=0, ddof=0, nan_policy='omit')
difflabels = ['diff2', 'diff4', 'diff8', 'diff12']
diffcols = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641']


#plot just evoked pupil response
fig = plt.figure(figsize = [4,3])
ax = fig.add_subplot(111)
ax.plot(times, gave_mean, color = '#000000', lw = 1)
ax.fill_between(times, y1=np.subtract(gave_mean, gave_sem), y2 = np.add(gave_mean, gave_sem),
                color = '#000000', edgecolor = None, lw = 0, alpha = 0.3)
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
ax.axhline(0, lw = 1, ls='dashed', color = '#000000')

#plot each single subject
fig = plt.figure(figsize = [4,3])
ax = fig.add_subplot(111)
ax.plot(times, gave.T, label=subs)
ax.legend()





fig = plt.figure(figsize = [8,3])
ax = fig.add_subplot(121)
ax.plot(times, corr_mean, color = '#31a354', lw = 1, label='correct')
ax.fill_between(times, y1 = np.subtract(corr_mean, corr_sem), y2 = np.add(corr_mean, corr_sem),
                color = '#31a354', edgecolor=None, lw=0, alpha=0.3)
ax.plot(times, incorr_mean, color = '#f03b20', lw = 1, label='incorrect')
ax.fill_between(times, y1 = np.subtract(incorr_mean, incorr_sem), y2 = np.add(incorr_mean, incorr_sem),
                color = '#f03b20', edgecolor=None, lw=0, alpha=0.3)
ax.legend(loc='lower left')
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
ax.axhline(0, lw = 1, ls='dashed', color = '#000000')

ax = fig.add_subplot(122)
ax.plot(times, diff_mean, color = '#3182bd', lw = 1, label='difference')
ax.fill_between(times, y1 = np.subtract(diff_mean, diff_sem), y2 = np.add(diff_mean, diff_sem),
                color = '#3182bd', edgecolor=None, lw=0, alpha=0.3)
ax.legend(loc='lower left')
ax.axvline(0, lw = 1, ls = 'dashed', color = '#000000')
ax.axhline(0, lw = 1, ls='dashed', color = '#000000')


fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.plot(times, difficulties_mean.T, label = difflabels)
for i in range(len(thetas)):
    ax.fill_between(times, y1 = np.subtract(difficulties_mean[i], difficulties_sem[i]).T, y2 = np.add(difficulties_mean[i], difficulties_sem[i]),
                    color = diffcols[i], edgecolor = None, lw = 0, alpha = 0.3)
ax.legend(loc='lower left')


    
    
    
    
    