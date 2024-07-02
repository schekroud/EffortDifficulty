#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:38:15 2024

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

_, binmids, binstarts, binends = TuningCurveFuncs.createFeatureBins(binstep, binwidth)

alldata = np.zeros(shape = [subs.size, binmids.size, ntimes]) * np.nan
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
    
    data = data * -1 #invert this, so more positive (larger) values = closer (mahalnobis distances are small when test is close to train)
    meandata = np.nanmean(data, axis=0)
    alldata[subcount] = meandata
        

gmean_tc = np.nanmean(alldata, axis=0)
        
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(gmean_tc, aspect = 'auto', origin = 'lower', cmap = 'RdBu_r', interpolation='none',
          extent = [times.min(), times.max(), binmids.min(), binmids.max()])
        

gmean_demean = gmean_tc.copy()

for tp in range(ntimes):
    isnan = np.isnan(gmean_demean[:,tp])
    imean = np.nanmean(gmean_demean[:,tp])
    gmean_demean[~isnan, tp] = np.subtract(gmean_demean[~isnan,tp], imean)
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(gmean_demean, aspect = 'auto', origin = 'lower', cmap = 'RdBu_r', interpolation='none',
          extent = [times.min(), times.max(), binmids.min(), binmids.max()])
            

fig = plt.figure(figsize = [15, 3])
ax = fig.add_subplot(141)
tinds = np.logical_and(np.greater_equal(times, -0.5), np.less_equal(times, -0.2))
tmpdat = np.nanmean(gmean_tc[:,tinds], axis=1) #average across time
ax.plot(binmids, tmpdat)
ax.set_xlabel('orientation')
ax.set_title('mean distances 0.5-0.2s prestim')
# ax.set_ylim([-0.15, 0.15])

ax = fig.add_subplot(142)
tinds = np.logical_and(np.greater_equal(times, 0.1), np.less_equal(times, 0.3))
tmpdat = np.nanmean(gmean_tc[:,tinds], axis=1) #average across time
ax.plot(binmids, tmpdat)
ax.set_xlabel('orientation')
ax.set_title('mean distances 0.1-0.3s post-stim')
# ax.set_ylim([-0.15, 0.15])

ax = fig.add_subplot(143)
tinds = np.logical_and(np.greater_equal(times, 0.4), np.less_equal(times, 0.7))
tmpdat = np.nanmean(gmean_tc[:,tinds], axis=1) #average across time
ax.plot(binmids, tmpdat)
ax.set_xlabel('orientation')
ax.set_title('mean distances 0.4-0.7s post-stim')
# ax.set_ylim([-0.15, 0.15])

ax = fig.add_subplot(144)
tinds = np.logical_and(np.greater_equal(times, 0.8), np.less_equal(times, 1.2))
tmpdat = np.nanmean(gmean_tc[:,tinds], axis=1) #average across time
ax.plot(binmids, tmpdat)
ax.set_xlabel('orientation')
ax.set_title('mean distances 0.8-1.2s post-stim')
# ax.set_ylim([-0.15, 0.15])

fig.tight_layout()
        
        
        