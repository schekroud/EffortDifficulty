#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:33:39 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import sklearn as skl
from sklearn import *
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib

import progressbar
progressbar.streams.flush()

loc = 'laptop'
if loc == 'laptop':
    sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
elif loc == 'workstation':
    sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

from funcs import getSubjectInfo
from decoding_functions import run_decoding, run_decoding_predproba
from joblib import parallel_backend
from TuningCurveFuncs import makeTuningCurve, createFeatureBins, visualise_FeatureBins


def wrap(x):
    return (x+180)%360 - 180
def wrap90(x):
    return (x+90)%180 - 90


os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

import progressbar
progressbar.streams.flush()

use_vischans_only  = True
use_pca            = False #using pca after selecting only visual channels is really bad for the decoder apparently
smooth_singletrial = True


# the distance between, and width of bins, shapes how many bins we have,
# and we need this info for preallocating arrays, lets do this here
# set up orientation bins

binstep = 4
binwidth = 11 #if you don't want overlap between bins, binwidth should be exactly half the binstep
# binstep, binwidth = 6, 3
nbins, binmids, binstarts, binends = createFeatureBins(binstep = binstep, binwidth = binwidth)
thetas = np.cos(np.radians(binmids))

visualise_bins = False
if visualise_bins:
    visualise_FeatureBins(binstarts, binmids, binends)

#params to align stim1locked and stim2locked data
srate = 100
tmin, tmax = -0.5, 1.5

for i in subs:
    sub   = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    print(f'\n- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'),
                                verbose = 'ERROR', preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = tmin, tmax = tmax) #crop for speed and relevance  
    epochs.resample(srate)
    times = epochs.times
    
    #read in stim2locked data and align
    epochs2 = mne.read_epochs(param['stim2locked'].replace('stim2locked', 'stim2locked_icaepoched_resamp_cleaned'))
    epochs2.drop_channels(dropchans)
    epochs2.crop(tmin = tmin, tmax = tmax)
    epochs2.resample(srate)
    
    
    if i == 16:
        epochs  = epochs[ 'blocknumber != 4']
        epochs2 = epochs2['blocknumber != 4'] 
    if i in [21, 25]:
        #these participants had problems in block 1
        epochs  = epochs[ 'blocknumber > 1']
        epochs2 = epochs2['blocknumber > 1']
    
    if use_vischans_only:
        #drop all but the posterior visual channels
        epochs = epochs.pick(picks = [
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                            'O1', 'Oz', 'O2'])
        epoch2s = epochs2.pick(picks = [
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                            'O1', 'Oz', 'O2'])
        
    data   = epochs._data.copy()  #get data matrix for stim1locked data
    data2  = epochs2._data.copy() #get data matrix for stim2locked data
    
    d = np.vstack([data, data2]) #concatenate data
    [ntrials, nfeatures, ntimes] = d.shape #get data shape -- [trials x channels x timepoints]
    
    
    if smooth_singletrial:
        # data  = sp.ndimage.gaussian_filter1d(data.copy(), sigma  = int(10 *  epochs.info['sfreq']/1000))
        # data2 = sp.ndimage.gaussian_filter1d(data2.copy(), sigma = int(10 * epochs2.info['sfreq']/1000)) 
        d = sp.ndimage.gaussian_filter1d(d.copy(), sigma = int(10 * srate/1000))
        #this is equivalent to smoothing with a 14ms gaussian blur as 500Hz sample rate. should help denoise a bit
        
    #demean data (stim1locked and stim2locked). this demeans the activity across sensors so it is 0-centred across the scalp
    demean_tpwise = True
    if demean_tpwise:
        for trl in range(ntrials):
            for tp in range(ntimes):
                # data[trl,:,tp]  = np.subtract( data[trl,:,tp],  data[trl,:,tp].mean())
                d[trl, :, tp] = np.subtract(d[trl,:,tp], d[trl,:,tp].mean())
                # data2[trl,:,tp] = np.subtract(data2[trl,:,tp], data2[trl,:,tp].mean())

    bdata1           = epochs.metadata.copy() 
    orientations1    = bdata.stim1ori.to_numpy()
    bdata2          = epochs2.metadata.copy()
    orientations2   = bdata2.stim2ori.to_numpy() #get orientation of stim2 ori each s2locked epoch refers to
    
    trls            = np.arange(ntrials) #index of trials
    trlinds = np.concatenate([np.ones(len(epochs)), np.add(np.ones(len(epochs2)),1)]).astype(int) #store whether an epoch belongs to a stim1 orientation or stim2 orientation
    
    orientations = np.concatenate([orientations1, orientations2])
    
    nruns = 2
    tuningCurve_all = np.empty([nruns, nbins, ntimes]) * np.nan #create empty array to store tuning curves
    tc_all = []    
    for irun in range(nruns):
        if irun == 0:
            trlsuse = np.less(orientations, 0) #take only left oriented trials
        elif irun == 1:
            trlsuse = np.greater(orientations, 0) #take only right oriented trials
        orisuse = orientations[trlsuse]
        datuse  = data[trlsuse]
        
        tc = np.zeros(shape = [orisuse.size, nbins, ntimes]) * np.nan
        for tp in range(ntimes):
            dists = makeTuningCurve(datuse[:,:,tp], orisuse,
                                    binstep = binstep, binwidth = binwidth,
                                    weight_trials = True)
            tc[:,:, tp] = dists
        tc_all.append(tc) #append tuning curve for this subset of trials
    
    #combine left and right for this subject
    tc_comb = np.vstack([tc_all[0], tc_all[1]])
    
    #really, we want to save the individual tuning curves to load in later so we dont need to constantly re-run this script
    #because all we did is subset and didn't change the order of trials, it should be simple to line up trials in tc_comb with trials in the task
    ltrls = bdata.query('stim1ori < 0').copy().reset_index(drop=True)
    rtrls = bdata.query('stim1ori > 0').copy().reset_index(drop=True)
    alltrls = pd.concat([ltrls, rtrls]) #contains trial metadata, where each row is a trial and matches each row in the trialwise tuning curve object
    
    #save data
    np.save(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}.npy'), tc_comb)
    
    if not op.exists(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')):
        alltrls.to_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv'), index=False)
    if i == 10:
        np.save(op.join(wd, 'data', 'tuningcurves', 'times.npy'), times) #save times