#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:36:58 2023

@author: sammi
"""
import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy
import copy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import pickle
import statsmodels as sm
import statsmodels.api as sma
%matplotlib

sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
import eyefuncs as eyes

wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)

eyedir = op.join(wd, 'data', 'eyes')

subs = [3, 4, 5, 6, 7]
subs = [3, 4,       7, 9, 10, 11]


correct     = []
incorrect   = []
alldata     = []
timepoint_logit_params = pd.DataFrame()
for isub in range(len(subs)):
    print('\n- - - - working on subject %s - - - - -\n'%(str(subs[isub])))
    sub = subs[isub]
    behfname = op.join(wd, 'data', 'datafiles', 'EffortDifficulty_s%02d_combined.csv'%sub)
    bdata = pd.read_csv(behfname)
    eyename = op.join(wd, 'data', 'eyes', 'preprocessed', 'EffDS%02da_preproc.pickle'%sub)
    with open(eyename, 'rb') as handle:
        subdat = pickle.load(handle) #load in the preprocessed data
    
    all_trigs = sorted(np.unique(subdat[0]['Msg'][:,2]))
    stim1trigs = dict(L = 'trig1', R = 'trig10', stim1trigs = ['trig1', 'trig10'])
    
    #get the epoched data
    stim1period = eyes.epoch(deepcopy(subdat),
                             trigger_values = stim1trigs['stim1trigs'],
                             traces = ['x', 'y', 'p', 'p_perc'],
                             twin = [-3, 6], srate = 1000)
    stim1period = eyes.apply_baseline(stim1period,
                                traces = ['x', 'y', 'p', 'p_perc'],
                                baseline_window = [-0.2, 0],
                                mode = 'mean', 
                                baseline_shift_gaze = [960, 540])
    
    #check how many nans there are across trials?
    ntrls = len(stim1period['info']['trigger'])
    nancheck = np.zeros(shape = (2,ntrls))
    for i in range(ntrls):
        nancheck[0,:] = np.sum(np.isnan(stim1period['p'][i,:]))
        nancheck[1,:] = np.sum(np.isnan(stim1period['p_perc'][i,:]))
            
    print('number of trials with nans in: '+str(nancheck.sum(axis=1)))
        
    #lowpass filter the data with a low pass cut off at 50hx

    traces_to_filt = ['p', 'p_perc']
    ds = deepcopy(stim1period['p_perc'])
    timerange = np.arange(stim1period['info']['tmin'], stim1period['info']['tmax'], step=1/stim1period['info']['srate'])
    # for trace in traces_to_filt:
    #     stim1period[trace] = eyes.lpfilter_epochs(stim1period, trace = trace, lp_cutoff = 30, srate = 1000)
        # stim1period[trace] = eyes.smooth(stim1period[trace], twin = 50, method = 'boxcar')
        
    #if you want to look at the effect of smoothing:
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(timerange, np.nanmean(ds, axis = 0), label = 'unfiltered epochs', lw = 1)
    # ax.plot(timerange, np.nanmean(stim1period['p'], axis=0), label = 'filtered epochs', lw = 1)
    # plt.legend(loc = 'lower left')
    
    alldata.append(stim1period['p_perc'])
    correctness = bdata['prevtrlfb'].to_numpy()
    correctness = np.select(
        [correctness == 'incorrect', correctness == 'correct', correctness == 'timed out'],
        [0, 1, -1], default = np.nan )
    #get data by correctness
    percept_corr = deepcopy(stim1period['p_perc'])[correctness == 1,:]
    percept_incorr = deepcopy(stim1period['p_perc'])[correctness == 0,:]
    
    #average across trials
    percept_corr = np.nanmean(percept_corr, axis = 0)
    percept_incorr = np.nanmean(percept_incorr, axis = 0)
    
    correct.append(percept_corr)
    incorrect.append(percept_incorr)
    
    plot_pupils = True
    if plot_pupils:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(timerange, percept_corr, label = 'prev trl correct', lw = 1)
        ax.plot(timerange, percept_incorr, label = 'prev trl incorrect', lw = 1)
        plt.legend(loc = 'lower left')
        fig.suptitle('ISI pupil dilation, subject '+ str(subs[isub]))
    
    
    #logistic regression per time point that looks at: P(correct) ~ angle difference + ISI + pupil dilation
    #need to get params for the trial and pupil dilation at that time point, fit model, save P(correct)
    run_timepoint_logistic = False
    logit_params = pd.DataFrame()
    if run_timepoint_logistic:
        for tp in range(timerange.size): #loop over timepoints
            pupils = stim1period['p_perc'][:,tp]
            pupils = stim1period['p'][:,tp]
            #zscore this (ignoring nans)
            pupils = eyes.nanzscore(pupils) #nans are set to zero so the model ignores them 
            desMat = deepcopy(bdata)[['PerceptDecCorrect','difficultyOri', 'delay1']]#, columns = ['accuracy', 'difficulty', 'ISI']]
            desMat = desMat.assign(pupilArea = pupils)
            desMat = desMat.query('PerceptDecCorrect != -1') #remove trials where they timed out
            acc = desMat['PerceptDecCorrect'].to_numpy()
            desMat = desMat.drop(labels = 'PerceptDecCorrect', axis = 1)
            desMat.columns = ['difficulty', 'ISI', 'pupil']
            desMat.insert(0, 'intercept', value = np.ones(len(desMat)).astype(int))
            
            tp_logit = sm.discrete.discrete_model.Logit(endog = acc,
                                                  exog = desMat);
            tp_fit = tp_logit.fit();
            tp_params = tp_fit.params.reset_index()
            tp_params.columns = ['regressor', 'beta']
            tp_params = tp_params.assign(timepoint = tp)
            logit_params = pd.concat([logit_params, tp_params])
    logit_params = logit_params.assign(subid = sub)
    timepoint_logit_params = pd.concat([timepoint_logit_params, logit_params])
    
    plot_timepoint_regression = False
    if plot_timepoint_regression:
        fig = plt.figure()
        fig.suptitle('pupil regressor over time for subject '+str(subs[isub]))
        ax = fig.add_subplot(111)
        ax.plot(timerange,logit_params.query('regressor == "pupil"').beta.to_numpy(), lw = 1, color = '#000000')
        ax.set_xlabel('time relative to stim1 onset (s)')
        ax.set_ylabel('pupil regressor beta (AU)')
        ax.vlines([0, 0.5], ls = 'dashed', lw = 1, ymin = -0.25, ymax = 0.2, color = '#bdbdbd')
        ax.hlines(y=0, xmin = timerange.min(), xmax = timerange.max(), ls = 'dashed', color = '#bdbdbd', lw = 1)
        
#plt.close('all')
#%%
#imshow to look at nans in the data
fig = plt.figure()
for isub in range(len(alldata)):
    ax = fig.add_subplot(2, 3, isub+1)
    ax.imshow(alldata[isub], aspect=  'auto', cmap = 'viridis', interpolation = 'none',
              extent = [timerange.min(), timerange.max(), 0, ntrls])
    ax.vlines(x = 0, ymin = 0, ymax = 320, color = '#000000', linestyles = 'dashed', lw = 1)
    ax.set_title('s%02d'%(subs[isub]))

#%%
# grand average effect?
correct = np.array(correct)
incorrect = np.array(incorrect)
diff = np.subtract(correct, incorrect)


correct_mean = np.nanmean(correct, axis = 0)
incorrect_mean = np.nanmean(incorrect, axis = 0)
diff_mean = np.nanmean(diff, axis = 0)
correct_sem = sp.stats.sem(correct, axis = 0)
incorrect_sem = sp.stats.sem(incorrect, axis = 0)
diff_sem = sp.stats.sem(diff, axis=0)

t_diff, t_diffp = sp.stats.ttest_1samp(diff, axis = 0, popmean = 0)

fig2 = plt.figure(figsize = (12,5))
ax = fig2.add_subplot(121)
ax.plot(timerange, diff_mean, label = 'difference', lw = 1, color = '#bdbdbd')
ax.plot(timerange, incorrect_mean, label = 'prev trl incorrect', lw = 1, color = '#fdc086')
ax.plot(timerange, correct_mean, label = 'prev trl correct', lw = 1, color = '#7fc97f')
ax.fill_between(x = timerange,
                y1 = np.subtract(diff_mean, diff_sem),
                y2 = np.add(diff_mean, diff_sem),
                color = '#bdbdbd', alpha = .1)
ax.fill_between(x = timerange,
                y1 = np.subtract(incorrect_mean, incorrect_sem),
                y2 = np.add(incorrect_mean, incorrect_sem),
                color = '#fdc086', alpha = .1)
ax.fill_between(x = timerange,
                y1 = np.subtract(correct_mean, correct_sem),
                y2 = np.add(correct_mean, correct_sem),
                color = '#7fc97f', alpha = .1)
ax.vlines(x = [0, 0.5], ls = 'dashed', color = '#000000', lw = 1, ymin = -1.5, ymax =1.5)
ax.hlines(y=0, xmin = timerange.min(), xmax = timerange.max(), ls = 'dashed', color = '#000000', lw =1 )
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('pupil diameter (AU)')
ax.set_title('grand mean difference')
ax.legend(loc = 'lower left')
ax = fig2.add_subplot(122)
ax.plot(timerange, t_diff, lw = 2, color = '#bdbdbd')
ax.hlines(y = 0, xmin = timerange.min(), xmax = timerange.max(), ls = 'dashed', color = '#000000', lw = 1)
ax.vlines(x = [0, 0.5], ls = 'dashed', color = '#000000', lw = 1, ymin = -3, ymax = 1.5)
ax.set_title('t-stat for difference')
fig2.tight_layout()
#%%

#plot single subjects and grand mean in one big figure
nsubs = len(subs)
fig = plt.figure(figsize=(10,10))
for sub in range(nsubs):
    ax = fig.add_subplot(5,4,sub+1)
    ax.plot(timerange, correct[sub,:], label = 'correct', color = '#66c2a5', lw = 2)
    ax.plot(timerange, incorrect[sub,:], label = 'incorrect', color = '#fc8d62', lw = 2)
    ax.set_xlabel('time rel. to s1 onset (s)')
    ax.set_ylabel('pupil area (AU)')
    ax.set_title('subject '+str(subs[sub]))
    ax.legend(loc='best', frameon = False)
ax = fig.add_subplot(5,4,17)
ax.plot(timerange, diff_mean, label = 'difference', lw = 2, color = '#66c2a5')
# ax.plot(timerange, incorrect_mean, label = 'incorrect', lw = 1, color = '#696969')
ax.fill_between(x = timerange,
                y1 = np.subtract(diff_mean, diff_sem),
                y2 = np.add(diff_mean, diff_sem),
                color = '#66c2a5', alpha = .1)
ax.vlines(x = [0, 0.5], ls = 'dashed', color = '#000000', lw = 1, ymin = -1.5, ymax = 1.5)
ax.hlines(y=0, xmin = timerange.min(), xmax = timerange.max(), ls = 'dashed', color = '#000000', lw =1 )
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('pupil area (AU)')
ax.set_title('grand average difference (correct - incorrect)')
fig.tight_layout()
