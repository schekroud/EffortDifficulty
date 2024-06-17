#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:52:50 2024

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
progressbar.streams.wrap_stderr()
# bar = progressbar.ProgressBar(max_value = len(subs)).start()

#set params for what file to load in per subject
binstep  = 4 
binwidth = 11
# binstep, binwidth = 6, 3
nparams  = 3 #number of parameters that we'll model the tuning curve

times = np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy'))
# times = np.arange(-0.5, 1.5, step = 0.01)
ntimes = times.size
paramfits_all = np.zeros(shape = [subs.size, nparams, ntimes]) * np.nan #for storing across-trial parameter estimates for each participant

# if op.exists(op.join(wd, 'data', 'tuningcurves', f'allsubs_paramfits_binstep{binstep}_binwidth{binwidth}.npy')):
#     paramfits_all = np.load(op.join(wd, 'data', 'tuningcurves', f'allsubs_paramfits_binstep{binstep}_binwidth{binwidth}.npy'))
# else:
subcount = -1

d1m = np.zeros(shape = [subs.size, 3, ntimes]) * np.nan
d2m = np.zeros_like(d1m) * np.nan
d3m = np.zeros_like(d1m) * np.nan
for i in subs:
    subcount +=1
    d1 = np.load(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_curvefit_binstep{binstep}_binwidth{binwidth}.npy'))
    d2 = np.load(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_fminsearch_binstep{binstep}_binwidth{binwidth}.npy'))
    d3 = np.load(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_curvefitinit_binstep{binstep}_binwidth{binwidth}.npy'))
    
    d1m[subcount] = np.nanmean(d1, axis=0)
    d2m[subcount] = np.nanmean(d2, axis=0)
    d3m[subcount] = np.nanmean(d3, axis=0)

#%%

m1 = np.nanmean(d1m, axis=0)
m2 = np.nanmean(d2m, axis=0)
m3 = np.nanmean(d3m, axis=0)

s1 = sp.stats.sem(d1m, axis=0, ddof=0, nan_policy='omit')
s2 = sp.stats.sem(d2m, axis=0, ddof=0, nan_policy='omit')
s3 = sp.stats.sem(d3m, axis=0, ddof=0, nan_policy='omit')



fig = plt.figure()
ax = fig.add_subplot(311)
ax.set_title('$\\beta_0$')
ax.plot(times, m1[0], color = 'b', lw = 1, label = 'curvefit')
ax.fill_between(times, np.subtract(m1[0], s1[0]), np.add(m1[0], s1[0]), color='b', alpha=0.3, edgecolor=None)
ax.plot(times, m2[0], color = 'r', lw = 1, label = 'fminsearch')
ax.fill_between(times, np.subtract(m2[0], s2[0]), np.add(m2[0], s2[0]), color='r', alpha=0.3, edgecolor=None)
ax.plot(times, m3[0], color = 'g', lw = 1, label = 'curvefit init')
ax.fill_between(times, np.subtract(m3[0], s3[0]), np.add(m3[0], s3[0]), color='g', alpha=0.3, edgecolor=None)
ax.legend(loc = 'upper left')
        
ax = fig.add_subplot(312)
ax.set_title('$\\beta_1$')
ax.plot(times, m1[1], color = 'b', lw = 1, label = 'curvefit')
ax.fill_between(times, np.subtract(m1[1], s1[1]), np.add(m1[1], s1[1]), color='b', alpha=0.3, edgecolor=None)
ax.plot(times, m2[1], color = 'r', lw = 1, label = 'fminsearch')
ax.fill_between(times, np.subtract(m2[1], s2[1]), np.add(m2[1], s2[1]), color='r', alpha=0.3, edgecolor=None)
ax.plot(times, m3[1], color = 'g', lw = 1, label = 'curvefit init')
ax.fill_between(times, np.subtract(m3[1], s3[1]), np.add(m3[1], s3[1]), color='g', alpha=0.3, edgecolor=None)
ax.legend(loc = 'upper left')

ax = fig.add_subplot(313)
ax.set_title('$\\alpha$')
ax.plot(times, m1[2], color = 'b', lw = 1, label = 'curvefit')
ax.fill_between(times, np.subtract(m1[2], s1[2]), np.add(m1[2], s1[2]), color='b', alpha=0.3, edgecolor=None)
ax.plot(times, m2[2], color = 'r', lw = 1, label = 'fminsearch')
ax.fill_between(times, np.subtract(m2[2], s2[2]), np.add(m2[2], s2[2]), color='r', alpha=0.3, edgecolor=None)
ax.plot(times, m3[2], color = 'g', lw = 1, label = 'curvefit init')
ax.fill_between(times, np.subtract(m3[2], s3[2]), np.add(m3[2], s3[2]), color='g', alpha=0.3, edgecolor=None)
ax.legend(loc = 'upper left')
        

#%%


fig = plt.figure(figsize = [18, 6])
plotnum = [1,4,7]
title = ['$\\beta_0$', '$\\beta_1$', '$\\alpha$']
for iparam in range(3):
    ax = fig.add_subplot(3, 3, plotnum[iparam])
    ax.plot(times, m1[iparam], color = 'b', lw = 1)
    ax.fill_between(times, np.add(m1[iparam], s1[iparam]), np.subtract(m1[iparam], s1[iparam]), alpha = 0.3, color = 'b', edgecolor = None)
    ax.set_title('curvefit ' + title[iparam])
    ax.set_xlim([-0.3, 1.3])

plotnum = [2, 5, 8]
for iparam in range(3):
    ax = fig.add_subplot(3, 3, plotnum[iparam])
    ax.plot(times, m2[iparam], color = 'r', lw = 1)
    ax.fill_between(times, np.add(m2[iparam], s2[iparam]), np.subtract(m2[iparam], s2[iparam]), alpha = 0.3, color = 'r', edgecolor = None)
    ax.set_title('fminsearch ' + title[iparam])
    ax.set_xlim([-0.3, 1.3])
    
plotnum = [3, 6, 9]
for iparam in range(3):
    ax = fig.add_subplot(3, 3, plotnum[iparam])
    ax.plot(times, m3[iparam], color = 'g', lw = 1)
    ax.fill_between(times, np.add(m3[iparam], s3[iparam]), np.subtract(m3[iparam], s3[iparam]), alpha = 0.3, color = 'g', edgecolor = None)
    ax.set_title('curvefit init: ' + title[iparam]) 
ax.set_xlim([-0.3, 1.3])
fig.tight_layout()
























