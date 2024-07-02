#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:40:17 2024

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
# bar = progressbar.ProgressBar(max_value = len(subs)).start()

#set params for what file to load in per subject
binstep  = 4 
binwidth = 11

#if you want to crop and model reduced number of time points
crop_data = False
tmin, tmax = -0.3, 1.3
times = np.round(np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy')), decimals=2)
if crop_data:
    tinds = np.logical_and(np.greater_equal(times, tmin), np.less_equal(times, tmax))
    times = times[tinds]
ntimes = times.size
    
subcount = -1
for i in subs:
    subcount += 1
    print(f'working on ppt {subcount+1}/{subs.size}')
    for smooth_alphas in [True, False]:
        
        if not smooth_alphas:
            smooth_sigma = None
            addSmooth = ''
            print(f'modelling with raw alpha values')
        else:
            smooth_sigma = 3
            addSmooth = str(smooth_sigma)
            print(f'modelling with smoothed alpha values')
            #read in single subject data
        if not op.exists(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'fixedalpha_b1only', f's{i}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{addSmooth}_no_bzero.npy')):
            data = np.load(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}.npy'))
            bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv'))
            
            #crop data if you want to
            if crop_data:
                data = data[:,:,tinds]
            [ntrials, nbins, ntimes] = data.shape
            
            #get bin info
            _, binmids, binstarts, binends = TuningCurveFuncs.createFeatureBins(binstep, binwidth)
            binmidsrad = np.deg2rad(binmids) #get radian values for bin centres
            
            data = data * -1 #invert this, so more positive (lager) values = closer (mahalnobis distances are small when test is close to train)
            d = data.copy()
            
            for trl in range(ntrials):
                for tp in range(ntimes):
                    isnan = np.isnan(d[trl,:,tp])
                    imean = np.nanmean(d[trl,:,tp])
                    d[trl, ~isnan, tp] = np.subtract(d[trl, ~isnan, tp], imean)

            
            #first we z-score the single trial distances to have mean 0 and var 1 then fit a fixed cosine model (cos(alpha *  theta*))
            dz = sp.stats.zscore(data.copy(), axis=1, #zscore over distances axis
                                nan_policy = 'omit')
            
            print(f'estimating alpha on z-scored distances, per trial and timepoint')
            alphas = np.zeros(shape = [ntrials, ntimes]) * np.nan # we get one alpha value for cosine width across distances for each trial and time point
            for trl in range(ntrials):   #loop over trials
                for tp in range(ntimes): #and loop over time points
                    dztp = dz[trl,:,tp].copy() #get z-scored distances across bins
                    tpa = TuningCurveFuncs.fitAlpha(binmidsrad, dztp, bounds = ([0], [1])) #bound alpha between 0 and 1
                    alphas[trl,tp] = tpa[0]
        
            # smooth_alphas = False #alphas can be noisy, smooth the alpha value across time to help mitigate this
            if smooth_alphas:
                alphas = sp.ndimage.gaussian_filter1d(alphas, sigma =  smooth_sigma)
            
            print(f'estimating beta on demeaned distances, per trial and timepoint')
            betas = np.zeros(shape = [ntrials, ntimes]) * np.nan
            for trl in range(ntrials):
                for tp in range(ntimes):
                    
                    dtp = d[trl,:,tp].copy() #get demeaned (inverted) distances across bins
                    
                    tpfit = sp.optimize.curve_fit(TuningCurveFuncs.b1_cosine, #fitting a cosine model with fixed alpha and only B1, using demeaned distances
                                                  xdata = np.multiply(binmidsrad, alphas[trl, tp]), # pre-multiply thetas by our previously-estimated alpha
                                                  ydata = dtp,
                                                  p0 = [1], #initial guess is the mean distance + B1 scaling of 1
                                                  maxfev = 5000, method = 'trf', nan_policy='omit')[0]
                    betas[trl,tp] = tpfit[0]
                                                  
            #save the alphas and betas
            np.save(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'fixedalpha_b1only', f's{i}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}_no_bzero.npy'), arr = alphas)
            np.save(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'fixedalpha_b1only', f's{i}_ParamFits_Betas_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}_no_bzero.npy'), arr = betas)
