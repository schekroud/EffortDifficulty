#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:33:43 2024

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

#set params for what file to load in per subject
binstep  = 4 
binwidth = 11

#if you want to crop and model reduced number of time points
crop_data = True
tmin, tmax = -0.3, 1.3
times = np.round(np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy')), decimals=2)
if crop_data:
    tinds = np.logical_and(np.greater_equal(times, tmin), np.less_equal(times, tmax))
    times = times[tinds]
ntimes = times.size

smooth_alphas = True
#set up some strings to make sure that we read in the right info, if looking at data with alpha smoothing
if smooth_alphas:
    sigma = 3
elif not smooth_alphas:
    sigma = ''
    
alphas = np.zeros(shape = [subs.size, ntimes])    * np.nan
betas  = np.zeros(shape = [subs.size, 2, ntimes]) * np.nan

subcount = -1
for i in subs:
    subcount += 1
    print(f'nworking on ppt {subcount+1}/{subs.size}')
    
    a = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', f's{i}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{sigma}.npy'))
    b = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', f's{i}_ParamFits_Betas_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{sigma}.npy'))
    
    if crop_data:
        a = a[:,tinds]
        b = b[:,:,tinds]
    
    alphas[subcount] = np.nanmean(a, axis=0) #average across trials
    betas[subcount]  = np.nanmean(b, axis=0) #average across trials
    
    
#%%   
#visualise these parameter estimates across subjects
am = np.nanmean(alphas,0)
bm = np.nanmean(betas,axis=0)

asem = sp.stats.sem(alphas, axis=0, ddof=0, nan_policy='omit')
bsem = sp.stats.sem(betas, axis=0, ddof=0, nan_policy='omit')


fig = plt.figure(figsize = [12, 8])
ax = fig.add_subplot(3,1,1)
ax.plot(times, bm[0], lw = 1.5, color = '#3182bd', label = '$\\beta_0$')
ax.fill_between(times, np.add(bm[0], bsem[0]), np.subtract(bm[0], bsem[0]),
                edgecolor = None, color = '#3182bd', alpha = 0.3)
ax.set_title('$\\beta_0$'); #ax.set_ylim([-10,0])

ax = fig.add_subplot(3,1,2)
ax.plot(times, bm[1], #sp.ndimage.gaussian_filter1d(bm[1], 5),
        lw = 1.5, color = '#31a354', label = '$\\beta_1$')
ax.fill_between(times, np.add(bm[1], bsem[1]), np.subtract(bm[1], bsem[1]),
                edgecolor = None, color = '#31a354', alpha = 0.3)
ax.set_title('$\\beta_1$'); #ax.set_ylim([-1, 1])

ax = fig.add_subplot(3,1,3)
ax.plot(times, am, lw = 1.5, color = '#e34a33', label = '$\\alpha$')
ax.fill_between(times, np.add(am, asem), np.subtract(am, asem),
                edgecolor = None, color = '#e34a33', alpha = 0.3)
ax.set_title('$\\alpha$')
fig.suptitle(f'across subject parameter estimates')
fig.tight_layout()

#%%
b2 = betas.copy()
b2 = np.delete(b2, 29, axis=0)

plt.figure(); plt.plot(times, b2.mean(0)[1], lw = 2, color = 'k'); plt.plot(times, bm[1], lw = 1.5, color = 'r')

#plot single subjects?
#%%
fig = plt.figure()
ax=fig.add_subplot(311)
ax.plot(times, betas[:,0].T, alpha = 0.5, lw = 1.5)
ax.plot(times, betas[:,0].mean(0), lw = 2, color = 'k')

ax=fig.add_subplot(312)
ax.plot(times, betas[:,1].T, alpha = 0.5, lw = 1.5)
ax.plot(times, betas.mean(0)[1], lw = 2, color = 'k')
ax = fig.add_subplot(313)
ax.plot(times, alphas.T, alpha = 0.5, lw = 1.5)
ax.plot(times, alphas.mean(0), lw = 2, color = 'k')



