#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:26:25 2024

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
from TuningCurveFuncs import makeTuningCurve, createFeatureBins


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



#because the distance between, and width of bins, shapes how many bins we have, and we need this info for preallocating arrays, lets do this here
#set up orientation bins
# binstep  = 4 #advance along orientation in steps of
# binwidth = 11 #width of bins for tuning curve (relation to binstep specifies overlap)
# binwidth = int(binstep/2) #if you dont want to have any overlap between bins
# binstep, binwidth = 6, 3
binmids = np.arange(-90, 91, binstep)
binstarts, binends = np.subtract(binmids, binwidth), np.add(binmids, binwidth)
nbins = binstarts.size #get the number of bins

#preallocate vectors to store each individual subjects data
sub_tcs = np.zeros(shape = [subs.size, nbins, 200]) *np.nan
subs_trialwise_tc = []
# progressbar.streams.wrap_stderr()
# bar = progressbar.ProgressBar(max_value = len(subs)).start()
subcount = -1
for i in subs:
    subcount += 1
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
    
    thetas = np.cos(np.radians(binmids))

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
    np.save(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}.npy'), alltrls)
    
    
    tc_comb = np.nanmean(tc_comb, axis=0) #average across all trials, having modelled left and right orientations separately
    alltrls.to_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv'), index=False)
    if i == 10:
        np.save(op.join(wd, 'data', 'tuningcurves', 'times.npy'), times)
    
#%%
    
    
    sub_tcs[subcount] = tc_comb
    subs_trialwise_tc.append(tc_all)
    
tc_plot = np.nanmean(sub_tcs, axis=0) #average across subjects


#demean?
tcp_dm = tc_plot.copy()
for tp in range(ntimes):
    isnan = np.isnan(tcp_dm[:,tp])
    imean = np.nanmean(tcp_dm[:,tp])
    tcp_dm[~isnan,tp] = np.subtract(tcp_dm[~isnan, tp], imean)


fig = plt.figure()
ax=fig.add_subplot(111)
plot = ax.imshow(tcp_dm*-1, aspect = 'auto', origin = 'lower', interpolation = 'gaussian', cmap = 'RdBu_r',
          # vmin = , vmax = ,
          extent = [times.min(), times.max(), binmids.min(), binmids.max()])
fig.colorbar(plot)

fig = plt.figure();
ax = fig.add_subplot(111)
ax.plot(binmids, tcp_dm[:,75]*-1)
ax.plot(binmids, tcp_dm[:,25]*-1, color='r')
ax.set_xlabel('relative orientation of test trial to train trials')
ax.set_ylabel('similarity (mahalanobis distance * -1)')
# ax.legend(loc = 'upper center')


#%% see how we can get single trial fit of a model where: dist = B0 + B1*cos(alpha*theta)


#subs_trialwise_tc contains single subject, across trial, tuning curves
#this is a nested list: array[subjectnumber][left, right orientations][trials per orientation side]
tmp = subs_trialwise_tc[0].copy() #get first subject
tmpl = tmp[0]
tmpr = tmp[1]
tmpall = np.vstack([tmp[0], tmp[1]]) #combine left/right trials


#demean per time point for visualisation
tmpl_plot = np.nanmean(tmpl, 0)
tmpr_plot = np.nanmean(tmpr, 0)
tmpall_plot = np.nanmean(tmpall, 0)

for tp in range(ntimes):
    iml = np.nanmean(tmpl_plot[:,tp])
    imr = np.nanmean(tmpr_plot[:,tp])
    ima = np.nanmean(tmpall_plot[:,tp])
    
    isnanl = np.isnan(tmpl_plot[:,tp])
    isnanr = np.isnan(tmpr_plot[:,tp])
    isnanall = np.isnan(tmpall_plot[:,tp])
    
    tmpl_plot[~isnanl,tp] = np.subtract(tmpl_plot[~isnanl,tp], iml)
    tmpr_plot[~isnanr,tp] = np.subtract(tmpr_plot[~isnanr,tp], imr)
    tmpall_plot[~isnanall,tp] = np.subtract(tmpall_plot[~isnanall,tp], ima)



#plot these two to see if it makes a difference which side we are decoding?
fig = plt.figure(figsize = [12, 4]);
ax=fig.add_subplot(131)
p1 = ax.imshow(tmpl_plot*-1, aspect='auto', interpolation='none', cmap = 'RdBu_r', origin = 'lower',
               vmin = -0.5, vmax = 0.5,
          extent = [times.min(), times.max(), binmids.min(), binmids.max()])
fig.colorbar(p1)

ax2=fig.add_subplot(132)
p2 = ax2.imshow(tmpr_plot*-1, aspect='auto', interpolation='none', cmap = 'RdBu_r', origin = 'lower',
               vmin = -0.5, vmax = 0.5,
          extent = [times.min(), times.max(), binmids.min(), binmids.max()])
fig.colorbar(p2)

ax3=fig.add_subplot(133)
p3 = ax3.imshow(tmpall_plot*-1, aspect='auto', interpolation='none', cmap = 'RdBu_r', origin = 'lower',
               vmin = -0.5, vmax = 0.5,
          extent = [times.min(), times.max(), binmids.min(), binmids.max()])
fig.colorbar(p3)

#get average tuning curve shape for 0.1-0.5s
timeinds = np.logical_and(np.greater_equal(times, 0.1), np.less_equal(times, 0.5))
tinds2   = np.logical_and(np.greater_equal(times, -0.4), np.less_equal(times, -0.2))

tc_trange = np.nanmean(tmpall_plot[:, timeinds], axis=1) #average across time
tc_tr2    = np.nanmean(tmpall_plot[:, tinds2], axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(binmids, tc_trange, color = 'b', label='poststim')
ax.plot(binmids, tc_tr2, color='red', label = 'prestim')
ax.set_xlabel('relative orientation')
ax.set_ylabel('distance')
fig.legend()


def cosinefit(thetas, B0, B1, alpha):
    return B0 + (B1 * np.cos(alpha * thetas))


paramfits = np.zeros(shape = [len(tmptp), 3, ntimes]) #ntrials, 3 params to be fit
fitparams = []
fitted = []
for tp in range(ntimes): #loop over timepoints
    tmptp = tmpall[:,:,tp].copy()
    for itrl in range(len(tmptp)):
        del(fitparams)
        del(fitted)
        isnan = np.isnan(tmptp[itrl])
        imids = binmids[~isnan]
        idat  = tmptp[itrl, ~isnan] * -1 #we want to take negative distances so more positive = closer
        fitparams = sp.optimize.curve_fit(cosinefit, imids, idat, maxfev = 5000)[0]
        # sp.optimize.fmin(cosinefit)
        fitted = fitparams[0] + fitparams[1]*np.cos(fitparams[2]*imids)
        paramfits[itrl,:,tp] = fitparams

        
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(imids, idat, color = 'b', lw=2, label = 'data')
        # ax.plot(imids, fitted, color = 'r', lw = 1, label = 'fitted')

mean_fit = np.nanmean(paramfits, axis=0)
sem_fit = sp.stats.sem(paramfits, axis=0, ddof=0)

fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(times, mean_fit[0])
ax.set_title('param0')

ax2 = fig.add_subplot(312)
ax2.plot(times, mean_fit[1])
ax2.set_title('param1')

ax3 = fig.add_subplot(313)
ax3.plot(times, mean_fit[2])
ax3.set_title('param2')
fig.suptitle('across time params when fitting $\\beta_0 + \\beta_1 * cos(\\alpha * \\theta)$') #\cdot will give dot multiplication symbol
fig.tight_layout()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(times, mean_fit[0], lw = 2, color = 'b', label = 'param0')
# ax.plot(times, mean_fit[1], lw = 1, color = 'r', label = 'param1')
# fig.legend()


#fminsearch 






