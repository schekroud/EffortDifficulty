#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:41:42 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
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
import TuningCurveFuncs


os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

import progressbar
progressbar.streams.flush()

#set params for what file to load in per subject
binstep  = 4 
binwidth = 11
# binstep, binwidth = 6, 3
nparams  = 3 #number of parameters that we'll model the tuning curve

times = np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy'))
# times = np.arange(-0.5, 1.5, step = 0.01)
ntimes = times.size
paramfits_all = np.zeros(shape = [subs.size, nparams, ntimes]) * np.nan #for storing across-trial parameter estimates for each participant

if op.exists(op.join(wd, 'data', 'tuningcurves', f'allsubs_paramfits_binstep{binstep}_binwidth{binwidth}.npy')):
    paramfits_all = np.load(op.join(wd, 'data', 'tuningcurves', f'allsubs_paramfits_binstep{binstep}_binwidth{binwidth}.npy'))
else:
    subcount = -1
    for i in subs:
        subcount +=1
        print(f'working on ppt {subcount+1}/{subs.size}\n')
        
        #read in single subject data
        data = np.load(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}.npy'))
        bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv'))
        [ntrials, nbins, ntimes] = data.shape
        
        #get bin info
        _, binmids, binstarts, binends = TuningCurveFuncs.createFeatureBins(binstep, binwidth)
        binmidsrad = np.deg2rad(binmids) #get radian values for bin centres
        
        data = data * -1 #invert this, so more positive (lager) values = closer (mahalnobis distances are small when test is close to train)
        meandata = np.nanmean(data, axis=0)
        
        # to analyse this tuning curve, we want to see how well a cosine fits to our data
        # where there is orientation preference in the brain activity, data should fit a cosine well
        # this is because scalp activity will more closely resemble feature bins with nearby values than with more distant values
        # to test this, we are going to fit a a model to the mahalanobis distances for each trial and timepoint, across feature bins
        # the model we will fit is: y = B0 + B1 * (cos(alpha * theta))
        # B0 models the mean distance across bins
        # B1 models the amplitude of the cosine fit (how much preference there is for orientation)
        # alpha models the width of the cosine fit, or the variability in angle preference across bins (representation uncertainty)
        
        paramfits = np.zeros(shape = [ntrials, nparams, ntimes]) #ntrials, 3 params to be fit
        fitparams = []
        fitted = []
        for tp in range(ntimes): #loop over timepoints
            tpdata = data[:,:,tp].copy() #get data for this time point
            for itrl in range(ntrials): #for each trial, for this time point
                del(fitparams)
                del(fitted)
                
                fitparams, imids, _ = TuningCurveFuncs.getCosineFit(angles = binmidsrad, data = tpdata[itrl])
                fitted = fitparams[0] + fitparams[1]*np.cos(fitparams[2]*imids)
                paramfits[itrl,:,tp] = fitparams
    
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # ax.plot(imids, idat, color = 'b', lw=2, label = 'data')
                # ax.plot(imids, fitted, color = 'r', lw = 1, label = 'fitted')
    
        #get mean parameters and store
        paramfits_all[subcount] = np.nanmean(paramfits, axis=0) #average parameter estimates across trials and store
    
    np.save(op.join(wd, 'data', 'tuningcurves', f'allsubs_paramfits_binstep{binstep}_binwidth{binwidth}.npy'), paramfits_all)
#%%
# tmp = paramfits_all.copy();
# tmp = np.delete(paramfits_all, 29, axis=0)

mean_fit = np.nanmean(paramfits_all, axis=0)
sem_fit = sp.stats.sem(paramfits_all, axis=0, ddof=0)

fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(times, mean_fit[0])
ax.fill_between(times, np.subtract(mean_fit[0], sem_fit[0]), np.add(mean_fit[0], sem_fit[0]), alpha = 0.3, edgecolor=None)
ax.set_title('$\\beta_0$')

ax2 = fig.add_subplot(312)
ax2.plot(times, mean_fit[1])
ax2.fill_between(times, np.subtract(mean_fit[1], sem_fit[1]), np.add(mean_fit[1], sem_fit[1]), alpha = 0.3, edgecolor=None)
ax2.set_title('$\\beta_1$')

ax3 = fig.add_subplot(313)
ax3.plot(times, mean_fit[2])
ax3.fill_between(times, np.subtract(mean_fit[2], sem_fit[2]), np.add(mean_fit[2], sem_fit[2]), alpha = 0.3, edgecolor=None)
ax3.set_title('$\\alpha$')
fig.suptitle('across time params when fitting $\\beta_0 + \\beta_1 * cos(\\alpha * \\theta)$') #\cdot will give dot multiplication symbol
fig.tight_layout()

#%%

fig = plt.figure()
ax = fig.add_subplot(111)
for isub in range(subs.size):
    ax.plot(times, paramfits_all[isub, 1,:])

#%%
y1 = np.cos(binmidsrad)
y2 = np.cos(2*binmidsrad)
y3 = np.cos(0.5*binmidsrad)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(binmidsrad, y1, label = '1x', color = 'b')
ax.plot(binmidsrad, y2, label = '2x', color = 'orange')
ax.plot(binmidsrad, y3, label = 'x/2', color = 'red')
ax.legend(loc = 'lower left')

    