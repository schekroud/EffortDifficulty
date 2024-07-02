#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:13:08 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import pickle
import sklearn as skl
from sklearn import *
import mne
import statsmodels as sm
import statsmodels.api as sma
import seaborn as sns
%matplotlib
import glmtools as glm

import progressbar
progressbar.streams.flush()

loc = 'laptop'
if loc == 'laptop':
    sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
elif loc == 'workstation':
    sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)

import eyefuncs_v2 as eyes
from funcs import getSubjectInfo
import TuningCurveFuncs

os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35,             39]) #some subjects get removed from eyetracking analyses


#set up directories that we need
eyedir = op.join(wd, 'data', 'eyes')
trlcheckdir = op.join(eyedir, 'trialchecks', 'stim1locked')



#get info about the eeg-related data we're going to load in
binstep  = 4 
binwidth = 11
eegtimes = np.round(np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy')), decimals=2)
neegtimes = eegtimes.size

smooth_alphas = True
#set up some strings to make sure that we read in the right info, if looking at data with alpha smoothing
if smooth_alphas:
    sigma = 3
elif not smooth_alphas:
    sigma = ''
    
    
regbetas = np.zeros(shape = [subs.size, 1, neegtimes, 6000]) #subjects x parameters x eeg times x pupil times
subcount = -1
for i in subs:
    subcount += 1
    print(f'nworking on ppt {subcount+1}/{subs.size}')
    
    sub   = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    
    alpha = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', f's{i}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{sigma}.npy'))
    betas = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', f's{i}_ParamFits_Betas_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{sigma}.npy'))
    
    #tuning curves are estimated for clockwise and anticlockwise orientations separately, then concatenated into one array
    #we stored a version of the behavioural data that is reordered to account for this, so it correctly lines up where row 1 in the bdata is row 1 in the tuning curve
    #read this in
    eegbfname = op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')
    eegbdata = pd.read_csv(eegbfname)
    
    
    eyename  = op.join(eyedir, 'preprocessed', f'EffDS{i}_preprocessed.pickle')
    eyebname = op.join(wd, 'data', 'datafiles', 'combined', 'EffortDifficulty_s%02d_combined_py.csv'%i)
    trlnans_perc = np.load(file = op.join(trlcheckdir, f"s{i}_stim1locked_eyes_nancheck_perTrial.npy"))
    to_keep      = trlnans_perc < 30
    #load in behavioural data
    eyebdata = pd.read_csv(eyebname)
    beh_nblocks = len(eyebdata.blocknumber.unique())
    
    #load in the preprocessed eyetracking data
    with open(eyename, 'rb') as handle:
        data = pickle.load(handle) 
    eyelims = [-2, 4]
    pupil = eyes.epoch(data, tmin = eyelims[0], tmax = eyelims[1], triggers = ['trig1', 'trig10'], channel = 'pupil_transformed')
    pupil.metadata = eyebdata #assign the behavioural data to this
    pupil.metadata = pupil.metadata.assign(tokeep = to_keep)#store whether a trial should be kept based on eyetracking data missing
    
    #do eyetracking trial rejection
    #as we are just interested in the relationship between pupil size and the tuning curve, we can ignore behaviour for now
    
    if sub == 16:
        #drop block 4 as we lost the eye
        pupil.data     = pupil.data[~pupil.metadata.blocknumber.eq(4)]
        pupil.metadata = pupil.metadata.query('blocknumber != 4')
    
    if sub in [21, 25]:
        #drop first block
        pupil.data     = pupil.data[~pupil.metadata.blocknumber.eq(1)]
        pupil.metadata = pupil.metadata.query('blocknumber != 1')
    
    if sub == 26:
        #drop trial 192
        pupil.data     = pupil.data[~pupil.metadata.trlid.eq(192)]
        pupil.metadata = pupil.metadata.query('trlid != 192')
    
    
    
    eegtrls = eegbdata.trlid.to_numpy()
    eyetrls = pupil.metadata.trlid.to_numpy()
    trls = np.arange(len(pupil.metadata))+1 #create list indexing trials
    
    noteeg = trls[~np.isin(trls, eegtrls)]
    noteyes = trls[~to_keep]
    
    #get indices of trials to be reject from either the eeg or the eyetracking
    trls2rem = np.unique(np.concatenate([noteeg, noteyes]))
    
    eegkeep = eegtrls[~np.isin(eegtrls, trls2rem)] #these are the trlids that should be kept
    
    alpha = alpha[~np.isin(eegtrls, trls2rem)]
    betas = betas[~np.isin(eegtrls, trls2rem)]
    eegbdata = eegbdata.query('trlid not in @trls2rem')
    
    pupil.data = pupil.data[~np.isin(eyetrls, trls2rem)]
    pupil.metadata = pupil.metadata.query("trlid not in @trls2rem")
    
    
    # data are matched in size, but not in order.
    # because distance calculation happens separately for anticlockwise and clockwise trials
    # need to reorder the tuning curve data rows to match the pupil rows so data are aligned
    eegreorder = eegbdata.trlid.argsort().to_numpy()
    eegbdata = eegbdata.iloc[eegreorder] #reorder rows to align
    alpha = alpha[eegreorder,:]
    betas = betas[eegreorder,:,:]
    
    
    neyetimes = pupil.times.size
    #neegtimes has how many eeg samples
    bdata = pupil.metadata.copy() #the eegbdata and pupil data are now aligned, so should be the same
    
    outputbetas = np.zeros(shape = [1, neegtimes, neyetimes])* np.nan
    
    bar = progressbar.ProgressBar(max_value = neegtimes).start()
    for eegtp in range(neegtimes):
        bar.update(eegtp)
        ialpha = alpha[:,eegtp]
        ib0 = betas[:,0,eegtp]
        ib1 = betas[:,1,eegtp]
        for eyetp in range(neyetimes):
            ieye = pupil.data[:,eyetp]
            
            lm = sma.GLM(endog = ieye, exog = sma.add_constant(ib1), family = sma.families.Gaussian()).fit()
            # lm2 = sma.GLM(endog = ieye, exog = ialpha, family = sma.families.Gaussian()).fit()
            outputbetas[0, eegtp, eyetp] = lm.params[0] #constant from the model
            outputbetas[1, eegtp, eyetp] = lm.params[1] #slope from the model
    
    regbetas[subcount] = outputbetas #store single subject beta coefficients
    
    
    #%%
fig = plt.figure()
ax = fig.add_subplot(111)
plot = ax.imshow(outputbetas[0], aspect = 'auto', vmin = -0.5, vmax = 0.5, 
                 origin='lower', interpolation = 'none', cmap = 'RdBu_r',
          extent = np.array([pupil.times.min(), pupil.times.max(), eegtimes.min(), eegtimes.max()])
          )
ax.axvline(0, ls = 'dashed', lw = 0.5, color = 'k')
ax.axhline(0, ls = 'dashed', lw = 0.5, color = 'k')
ax.set_xlabel('pupil time')
ax.set_ylabel('eeg time')
fig.colorbar(plot)



tmp = outputbetas.copy()
tmp = tmp[:,:,np.logical_and(np.greater_equal(pupil.times, -0.5), np.less_equal(pupil.times, -0.2))]
tmp = np.nanmean(tmp, axis=-1) #average across this time window

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(eegtimes, tmp[0])

    
    