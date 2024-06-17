#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:27:03 2024

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
    # meandata = np.nanmean(data, axis=0)
    
    # to analyse this tuning curve, we want to see how well a cosine fits to our data
    # where there is orientation preference in the brain activity, data should fit a cosine well
    # this is because scalp activity will more closely resemble feature bins with nearby values than with more distant values
    # to test this, we are going to fit a a model to the mahalanobis distances for each trial and timepoint, across feature bins
    # the model we will fit is: y = B0 + B1 * (cos(alpha * theta))
    # B0 models the mean distance across bins
    # B1 models the amplitude of the cosine fit (how much preference there is for orientation)
    # alpha models the width of the cosine fit, or the variability in angle preference across bins (representation uncertainty)
    
    paramfits = np.zeros(shape = [ntrials, nparams, ntimes]) * np.nan #ntrials, 3 params to be fit
    paramfits2 = np.zeros_like(paramfits) * np.nan
    paramfits3 = np.zeros_like(paramfits) * np.nan
    fitparams = []
    
    fitparams2 = []
    fitparams3 = []
    
    bar = progressbar.ProgressBar(max_value = ntimes).start()
    for tp in range(ntimes): #loop over timepoints
        bar.update(tp)
        tpdata = data[:,:,tp].copy() #get data for this time point
        for itrl in range(ntrials): #for each trial, for this time point
            del(fitparams)
            # del(fitted)
            
            fitparams, imids, idat = TuningCurveFuncs.getCosineFit(angles = binmidsrad, data = tpdata[itrl], fitType = 'curve')
            # fitted = fitparams[0] + (fitparams[1]*np.cos(fitparams[2]*imids))
            
            fitparams2, imids2, idat2 = TuningCurveFuncs.getCosineFit(angles = binmidsrad, data = tpdata[itrl], fitType = 'fmin')
            # fitted2 = fitparams2[0] + (fitparams2[1]*np.cos(fitparams2[2]*imids2))
            
            fitparams3, imids3, idat3 = TuningCurveFuncs.getCosineFit(angles = binmidsrad, data = tpdata[itrl],
                                                                      fitType = 'curve', p0 = [np.nanmean(tpdata[itrl]), 1, 1])
            # fitted3 = fitparams3[0] + (fitparams3[1]*np.cos(fitparams3[2]*imids3))
            
            # fitparams3, imids3, idat3 = TuningCurveFuncs.getCosineFit(angles = binmidsrad, data = tpdata[itrl], fitType = 'curve',
            #                                                           p0 = [np.nanmean(tpdata[itrl]), 1, 1]) #set some initial guess params
            # fitted3 = fitparams3[0] + (fitparams3[1]*np.cos(fitparams3[2]*imids3))
            
            # fig = plt.figure(); ax = fig.add_subplot(111)
            # ax.plot(np.deg2rad(binmids), tpdata[itrl], 'k', lw = 2, label = 'raw')
            # ax.plot(imids, fitted, 'b', lw = 1, label = 'curvefit')
            # ax.plot(imids2, fitted2, 'r', lw = 1, label = 'fmin')
            # ax.plot(imids3, fitted3, 'green', lw  = 1, label = 'curvefit init')
            # ax.legend(loc = 'lower left')
            
            paramfits[itrl,:,tp] = fitparams
            paramfits2[itrl,:,tp] = fitparams2
            paramfits3[itrl,:,tp] = fitparams3
    
    #save paramfits as separate files to read in later:
    
    np.save(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_curvefit_binstep{binstep}_binwidth{binwidth}.npy'), paramfits)
    np.save(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_fminsearch_binstep{binstep}_binwidth{binwidth}.npy'), paramfits2)
    np.save(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_curvefitinit_binstep{binstep}_binwidth{binwidth}.npy'), paramfits3)
                
# np.save(op.join(wd, 'data', 'tuningcurves', f'allsubs_paramfits_binstep{binstep}_binwidth{binwidth}.npy'), paramfits_all)        
#%%                
subcount = -1
d1m = np.zeros(shape = [subs.size, 3, 200])
for i in subs:
    subcount +=1
    d1 = np.load(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_curvefit_binstep{binstep}_binwidth{binwidth}.npy'))
    d2 = np.load(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_fminsearch_binstep{binstep}_binwidth{binwidth}.npy'))
    d3 = np.load(op.join(wd, 'data', 'tuningcurves', 'paramfits', f's{i}_paramfits_curvefitinit_binstep{binstep}_binwidth{binwidth}.npy'))





#%%
m1 = paramfits.mean(0)
m2 = paramfits2.mean(0)
m3 = paramfits3.mean(0)

s1 = sp.stats.sem(paramfits,  axis=0, ddof=0, nan_policy='omit')
s2 = sp.stats.sem(paramfits2, axis=0, ddof=0, nan_policy='omit')
s3 = sp.stats.sem(paramfits3, axis=0, ddof=0, nan_policy='omit')

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
        
#%%
        
xdat = np.nanmean(meandata[:,70:90], axis=1)
        
        
isnan = np.isnan(xdat)
imids = np.deg2rad(binmids[~isnan])
moddat = xdat[~isnan]
mod_dmean = np.subtract(moddat, moddat.mean())

res = sp.optimize.fmin(func = TuningCurveFuncs.fmin_func, #function to minimize
                 x0 = [moddat.mean(), 1, 1], #initial guess for parameters
                 args = (moddat, imids))
res_demean = sp.optimize.fmin(func = TuningCurveFuncs.fmin_func, #function to minimize
                 x0 = [0, 1, 1], #initial guess for parameters
                 args = (mod_dmean, imids))

bf = res[0] + (res[1] * np.cos(res[2] * imids))
bf_dm = res_demean[0] + (res_demean[1] * np.cos(res_demean[2] * imids))

#visualise difference between modelling raw and modelling demeaned data
fig = plt.figure(figsize = (10,3));
ax = fig.add_subplot(121)
ax.plot(imids, moddat, 'b', lw=2, label = 'raw')
ax.plot(imids, bf, 'r', lw = 1, label = 'fmin')
ax.set_title('raw data model with fminsearch')
ax = fig.add_subplot(122)
ax.set_title('demeaned data model with fmin search')
ax.plot(imids, mod_dmean, 'b', lw = 2, label = 'demeaned')
ax.plot(imids, bf_dm, 'r', lw = 1, label = 'fmin')






res2    = TuningCurveFuncs.getCosineFit(imids, moddat)[0]
bestfit2 = res2[0] + (res2[1] * np.cos(res2[2] * np.deg2rad(imids)))

res3 = sp.optimize.curve_fit(f = TuningCurveFuncs.cosmodel,
                             xdata = np.deg2rad(imids),
                             ydata = moddat, 
                             p0 = [moddat.mean(), 1, 1])[0]
bf3 = res3[0] + (res3[1] * np.cos(res3[2] * np.deg2rad(imids)))



fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(imids, moddat,   lw = 2, color = 'black', label = 'true')
ax.plot(imids, bestfit,  lw = 2, color = 'red', label = 'fmin')
ax.plot(imids, bestfit2, lw = 1, color = 'green', label = 'curvefit')
ax.plot(imids, bf3,      lw = 1, color = 'blue', label = 'curvefit_init')
ax.legend(loc = 'lower center') 




        