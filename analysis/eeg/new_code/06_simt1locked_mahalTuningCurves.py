#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:25:46 2024

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
%matplotlib
from scipy.spatial.distance import mahalanobis
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
from TuningCurveFuncs import makeTuningCurve


def wrap(x):
    return (x+180)%360 - 180
def wrap90(x):
    return (x+90)%180 - 90


os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

figpath = op.join(wd, 'figures', 'decoding', 'LDA')

import progressbar
progressbar.streams.flush()

use_vischans_only  = True
use_pca            = False #using pca after selecting only visual channels is really bad for the decoder apparently
smooth_singletrial = True

# progressbar.streams.wrap_stderr()
# bar = progressbar.ProgressBar(max_value = len(subs)).start()
# subcount = -1
for i in subs:
    # subcount += 1
    # bar.update(subcount+1)
    # sub   = dict(loc = 'laptop', id = i)
    sub   = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    print(f'\n- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'),
                                verbose = 'ERROR', preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -0.5, tmax = 1.5) #crop for speed and relevance  
    epochs.resample(100)
    times = epochs.times
    
    # epochs = epochs['stim1ori < 0']
    # epochs = epochs['stim1ori > 0']
    
    if i == 16:
        epochs = epochs['blocknumber != 4']
    if i in [21, 25]:
        #these participants had problems in block 1
        epochs = epochs['blocknumber > 1']
    
    if use_vischans_only:
        #drop all but the posterior visual channels
        epochs = epochs.pick(picks = [
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                            'O1', 'Oz', 'O2'])
    data   = epochs._data.copy() #get the data matrix
    [ntrials, nfeatures, ntimes] = data.shape #get data shape -- [trials x channels x timepoints]
    
    if smooth_singletrial:
        data = sp.ndimage.gaussian_filter1d(data.copy(), sigma = int(10 * epochs.info['sfreq']/1000))
        #this is equivalent to smoothing with a 14ms gaussian blur as 500Hz sample rate. should help denoise a bit
        
    #demean data
    demean_tpwise = True
    if demean_tpwise:
        for trl in range(ntrials):
            for tp in range(ntimes):
                data[trl,:,tp] = np.subtract(data[trl,:,tp], data[trl,:,tp].mean())
    # fig = plt.figure();
    # ax = fig.add_subplot(111)
    # for trl in range(ntrials):
    #     ax.plot(epochs.times, data[trl].mean(0))
    
    bdata  = epochs.metadata.copy() 

    orientations = bdata.stim1ori.to_numpy()
    
    trls = np.arange(ntrials) #index of trials
    #set up orientation bins
    binstep  = 4 #advance along orientation in steps of
    binwidth = 11 #width of bins for tuning curve (relation to binstep specifies overlap)
    # binwidth = int(binstep/2) #if you dont want to have any overlap between bins
    binmids = np.arange(-90, 91, binstep)
    binstarts, binends = np.subtract(binmids, binwidth), np.add(binmids, binwidth)
    
    
    visualise_bins = False
    if visualise_bins:
        nangles = np.arange(binstarts.min(), binends.max()+1, 1)
        nangles = np.deg2rad(nangles) #conv to radians
        tmpcurves = np.zeros(shape = [binstarts.size, nangles.size]) * np.nan #create empty thing to populate to show angle curves for bins
        tmpbins   = np.zeros_like(tmpcurves)
        binangs   = np.zeros(shape = [binstarts.size, binwidth*2]) #store angles for each bin
        for ibin in range(binstarts.size):
            istart, iend = binstarts[ibin], binends[ibin] #get the angles
            istartrad, iendrad = np.deg2rad(binstarts[ibin]), np.deg2rad(binends[ibin])
            iangles = np.arange(istart, iend)
            binangs[ibin] = iangles
            # iangles = np.deg2rad(iangles)
            imid = binmids[ibin]
            imidrad = np.deg2rad(binmids[ibin])
            # ibinmask = np.isin(nangles, np.arange(istart, iend)).astype(int)
            ibinmask = np.logical_and(np.greater_equal(nangles, istartrad), np.less(nangles, iendrad))
            tmpbins[ibin] = ibinmask
            ianglesrad = np.deg2rad(iangles)
            # tmp  = np.cos(np.radians(((iangles-imid)/binwidth*2))* np.pi)
            tmp = np.cos( ((ianglesrad-imidrad)/np.deg2rad(binwidth*2)*np.pi) )
            icurve = ibinmask.copy().astype(float) * np.nan;
            icurve[ibinmask==1] = tmp
            tmpcurves[ibin] = icurve
        
        #visualise the weightings that will be applied to different angles within each bin
        fig = plt.figure(); ax = fig.add_subplot(111)
        for ibin in range(binstarts.size):
            ax.plot(np.deg2rad(binangs[ibin]), np.squeeze(tmpcurves[ibin, np.where(~np.isnan(tmpcurves[ibin]))])  )
        
        #visualise what angles are possible in each bin
        fig = plt.figure(); ax = fig.add_subplot(111)
        # # ax.plot(nangles, sp.ndimage.gaussian_filter1d(tmpbins,3).T)
        ax.plot(nangles, tmpbins.T)
    
    
    nbins = binstarts.size
    thetas = np.cos(np.radians(binmids))
    

    nruns = 2
    tuningCurve = np.empty([nruns, nbins, ntimes]) * np.nan #create empty array to store tuning curves
    for irun in np.arange(nruns):
        if irun == 0:
            #only look at left orientation trials
            print('decoding left orientation trials')
            touse = np.less(orientations, 0)
        elif irun == 1:
            #only look at right orientation trials
            print('decoding right orientation trials')
            touse = np.greater(orientations,0)
            
        oris_use = orientations[touse]
        datause = data.copy()[touse,:,:]
        trls = np.arange(touse.sum())
        
        dists = np.zeros(shape = [trls.size, nbins, ntimes]) * np.nan
        d = np.zeros([trls.size, nbins, ntimes])       * np.nan #each trial has a distance from the other bins, at each time point
        
        progressbar.streams.wrap_stderr()
        bar = progressbar.ProgressBar(max_value = len(trls)+1, min_value=1).start()
        for trl in trls:
            bar.increment()
            xtrain   = datause[np.setdiff1d(trls, trl), :, :] #get training data
            xtest    = datause[trl, :, :]
            oritrain = oris_use[np.setdiff1d(trls, trl)]
            oritest  = oris_use[trl] 
            
            reloris = wrap90(np.subtract(oritrain, oritest))
            relorisrad = np.deg2rad(reloris)
            m     = np.zeros([nbins, nfeatures, ntimes]) * np.nan #store mean activity across trials in each bin
            inbin = np.zeros([len(xtrain), nbins])       * np.nan
            
            for tp in range(ntimes): #loop over time
                tpxtrain = xtrain[:,:,tp]
                tpxtest  = xtest[:,tp]
                
                icov = np.linalg.pinv(np.cov(tpxtrain.T))
                mahaldist  = skl.metrics.DistanceMetric.get_metric('mahalanobis', VI = icov)
                
                #loop over bins, get weighted activity in each bin
                m = np.zeros([nbins, nfeatures]) * np.nan
                for b in range(nbins):
                    istart, iend, imid = np.radians(binstarts[b]), np.radians(binends[b]), np.radians(binmids[b]) #get all angles in radians
                    
                    # binrange = np.arange(istart, iend +1)
                    # if binwidth == int(binstep/2):
                    #     binrange = np.arange(istart, iend) #non-overlapping bins here if specified earlier
                    # trlcheck = np.isin(reloris, binrange) #find trials where the relative orientation of the test trial is in the range of orientations we want
                    
                    if binwidth == int(binstep/2): #check if non-overlapping bins are required
                        trlcheck = np.logical_and(np.greater_equal(relorisrad, istart), np.less(relorisrad, iend))
                    else:
                        trlcheck = np.logical_and(np.greater_equal(relorisrad, istart), np.less_equal(relorisrad, iend))
                    
                    #get relative orientations for trials in the bin
                    binoris = relorisrad[trlcheck] #get the relative orientation of the trials that fit into the bin of interest
                    bindata = tpxtrain[trlcheck] #get data for the trials that fit into the bin of interest
                    
                    weight_trials = True
                    if weight_trials:
                        w = np.cos( ((binoris-imid)/np.radians(binwidth*2))* np.pi) #cosine weighting based on distance from the centre of the bin
                    else:
                        w = np.ones(binoris.size) #just multiply everything by 1 to keep the same (no weighting)
                    
                    bindata =  np.multiply(bindata, w.reshape(-1, 1)) #reshape weights to allow the multiplication to happen
                    #note that where we are not weighting trials based on distance from bincentre, we just multiply by 1 (to keep patterns the same)
                    if bindata.size > 0: #this just stops it churning out warnings that get annoying (where no data in a bin because the relative oris dont exist)
                        m[b,:] = np.nanmean(bindata, axis=0) #average patterns across trials to get average pattern across sensors for this orientation bin
                
                nanbins = np.isnan(m).sum(axis=1)>0
                idists = mahaldist.pairwise(tpxtest.reshape(1,-1), m[~nanbins])
                d[trl, ~nanbins, tp] = idists
        
        #d now contains the mahalanobis distance between the test trial and binned trials of other relative orientations, for each time point
        #where d is small, the test trial is close to the average orientation of the reference set. Where this value is large, it is far away (dissimilar)
        # we can invert this (multiply by -1) so that larger (more positive) numbers = closer,
        # and smaller (more negative values) are further away (more dissimilar)
        d2 = d.copy()        
        d2 = np.multiply(d2, -1)            
        d2 = np.nanmean(d2, axis=0)
        
        d3 = d2.copy()
        #demean this across bins at each time point
        for tp in range(ntimes):
            d3[:,tp] = np.subtract(d3[:,tp], np.nanmean(d3[:,tp]))
            
        tuningCurve[irun] = d3
    
    tuningCurve_plot = np.nanmean(tuningCurve, axis=0) #average across left/right orientations to get average tuning curve
    # plt.figure(); plt.plot(binmids, d2[:,100]) #100 is the tp for .5s if at 100Hz srate
    
    d4 = tuningCurve_plot.copy()
    for tp in range(ntimes):
        d4[:,tp] = np.multiply(d4[:,tp], np.cos(np.radians(binmids)*np.pi))
    
    # plt.figure(); plt.plot(binmids, d4[:,500]) #100 is the tp for .5s if at 100Hz srate
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    plot = ax.imshow(tuningCurve_plot, origin = 'lower', aspect='auto', cmap = 'RdBu_r', interpolation='none',
              # vmin = -6, vmax = -3,
              #vmin = -3, vmax = 0.5,
                # vmin = -4.4, vmax = -3.8,
              extent = [times.min(), times.max(), binmids.min(), binmids.max()])
    # ax.set_ylim([-35, 35])
    fig.colorbar(plot)







#%%  

    