#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:01:40 2023

@author: sammi
"""
import os.path as op
import numpy as np
import scipy as sp
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt

def getSubjectInfo(subject, ):
    
    param = {}
    
    if subject['loc']   == 'workstation':
        param['path']   = '/home/sammirc/Desktop/postdoc/EffortDifficulty/data'
        wd              = '/ohba/pi/knobre/schekroud/postdoc/student_projects/EffortDifficulty'
    elif subject['loc'] == 'laptop': 
        param['path']   = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/data'
        wd              = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
    elif subject['loc'] == 'pc':
        param['path']   = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/data'
        wd              = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
    
    path = param['path']
    substr = 's%02d'%subject['id']
    eegpath = path[:-5] #remove /data  from pathstring 
    
    param['subid']          = substr 
    param['behaviour']      = op.join(path, 'datafiles', 'EffortDifficulty_s%02d_combined.csv'%subject['id']) #behavioural data file
    param['raweeg']         = op.join(wd, 'eeg', substr, 'EffortDifficulty_s%02d.dat'%subject['id']) # raw eeg data
    param['eeg_preproc']    = op.join(wd, 'eeg', substr, 'EffortDifficulty_s%02d_preproc-raw.fif'%subject['id']) #preprocessed data
    param['asc']            = op.join(path, 'eyes', 'asc', 'EffDS%02d.asc'%subject['id'])
    param['raweyes']        = op.join(path, 'eyes', 'raw', 'EffDS%02d_raw.pickle'%subject['id'])
    param['preproceyes']    = op.join(path, 'eyes', 'preprocessed', 'EffDS%02d_preproc.pickle'%subject['id'])
    
    #paths to save other eeg objects
    param['stim1locked']    = op.join(wd, 'eeg', substr, 'EffortDifficulty_s%02d_stim1locked-epo.fif'%subject['id'])
    param['stim2locked']    = op.join(wd, 'eeg', substr, 'EffortDifficulty_s%02d_stim2locked-epo.fif'%subject['id'])
    param['fblocked']       = op.join(wd, 'eeg', substr, 'EffortDifficulty_s%02d_fblocked-epo.fif'%subject['id'])
    
    
    # if subject['id'] == :
    #     param['badchans'] = []
    
    if subject['id'] == 10:
        param['badchans'] = ['AF7']
    if subject['id'] == 11:
        param['badchans'] = ['TP7']
    if subject['id'] == 12:
        param['badchans'] = ['AF3', 'T7', 'T8', 'TP7', 'TP8']
    if subject['id'] == 13:
        param['badchans'] = ['FT8', 'T7', 'T8']
    if subject['id'] == 14:
        param['badchans'] = []
    if subject['id'] == 15:
        param['badchans'] = []
    if subject['id'] == 16:
        param['badchans'] = ['T8']
    # if subject['id'] == 17:
    #     param['badchans'] = []
    # if subject['id'] == 18:
    #     param['badchans'] = []
    # if subject['id'] == 19:
    #     param['badchans'] = []
    # if subject['id'] == 20:
    #     param['badchans'] = []
    
        
    return param
    
def gesd(x, alpha = .05, p_out = .1, outlier_side = 0):
    import numpy as np
    import scipy.stats
    import copy

    '''
    Detect outliers using Generalizes ESD test
    based on the code from Romesh Abeysuriya implementation for OSL
      
    Inputs:
    - x : Data set containing outliers - should be a np.array 
    - alpha : Significance level to detect at (default = .05)
    - p_out : percent of max number of outliers to detect (default = 10% of data set)
    - outlier_side : Specify sidedness of the test
        - outlier_side = -1 -> outliers are all smaller
        - outlier_side = 0 -> outliers could be small/negative or large/positive (default)
        - outlier_side = 1 -> outliers are all larger
        
    Outputs
    - idx : Logicial array with True wherever a sample is an outlier
    - x2 : input array with outliers removed
    
    For details about the method, see
    B. Rosner (1983). Percentage Points for a Generalized ESD Many-outlier Procedure, Technometrics 25(2), pp. 165-172.
    http://www.jstor.org/stable/1268549?seq=1
    '''

    if outlier_side == 0:
        alpha = alpha/2


    if type(x) != np.ndarray:
        x = np.asarray(x)

    n_out = int(np.ceil(len(x)*p_out))

    if any(~np.isfinite(x)):
        #Need to find outliers only in non-finite x
        y = np.where(np.isfinite(x))[0] # these are the indexes of x that are finite
        idx1, x2 = gesd(x[np.isfinite(x)], alpha, n_out, outlier_side)
        # idx1 has the indexes of y which were marked as outliers
        # the value of y contains the corresponding indexes of x that are outliers
        idx = [False] * len(x)
        idx[y[idx1]] = True

    n      = len(x)
    temp   = x
    R      = np.zeros((1, n_out))[0]
    rm_idx = copy.deepcopy(R)
    lam    = copy.deepcopy(R)
    
    
    for j in range(0,int(n_out)):
        i = j+1
        if outlier_side == -1:
            rm_idx[j] = np.nanargmin(temp)
            sample    = np.nanmin(temp)
            R[j]      = np.nanmean(temp) - sample
        elif outlier_side == 0:
            rm_idx[j] = int(np.nanargmax(abs(temp-np.nanmean(temp))))
            R[j]      = np.nanmax(abs(temp-np.nanmean(temp)))
        elif outlier_side == 1:
            rm_idx[j] = np.nanargmax(temp)
            sample    = np.nanmax(temp)
            R[j]      = sample - np.nanmean(temp)

        R[j] = R[j] / np.nanstd(temp)
        temp[int(rm_idx[j])] = np.nan

        p = 1-alpha/(n-i+1)
        t = scipy.stats.t.ppf(p,n-i-1)
        lam[j] = ((n-i) * t) / (np.sqrt((n-i-1+t**2)*(n-i+1)))

    #And return a logical array of outliers
    idx = np.zeros((1,n))[0]
    idx[np.asarray(rm_idx[range(0,np.max(np.where(R>lam))+1)],int)] = np.nan
    idx = ~np.isfinite(idx)

    x2 = x[~idx]


    return idx, x2


def plot_AR(epochs, method = 'gesd', zthreshold = 1.5, p_out = .1, alpha = .05, outlier_side = 1):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import scipy.stats
    from matplotlib import pyplot as plt

    #Get data, variance, number of trials, and number of channels
    dat     = epochs.get_data()
    var     = np.var(dat, 2)
    ntrials = np.shape(dat)[0]
    nchan   = len(epochs.ch_names)

    #set up the axis for the plots
    x_epos  = range(1,ntrials+1)
    y_epos  = np.mean(var,1)
    y_chans = range(1,nchan+1)
    x_chans = np.mean(var,0)

    #scale the variances
    y_epos  = [x * 10**6 for x in y_epos]
    x_chans = [x * 10**6 for x in x_chans]

    #Get the zScore
    zVar = scipy.stats.zscore(y_epos)

    #save everything in the dataFrame
    df_epos           = pd.DataFrame({'var': y_epos, 'epochs': x_epos, 'zVar': zVar})
    df_chans          = pd.DataFrame({'var': x_chans, 'chans': y_chans})

    # Apply the artefact rejection method
    if method == 'gesd':
        try:
            idx,x2            = gesd(y_epos, p_out=p_out, alpha=alpha, outlier_side=outlier_side) #use the gesd to find outliers (idx is the index of the outlier trials)
        except:
            print('***** gesd failed here, no trials removed *****')
            idx = []
        keepTrials        = np.ones((1,ntrials))[0]
        keepTrials[idx]   = 0
        title = 'Generalized ESD test (alpha=' + str(alpha) + ', p_out=' + str(p_out) + ', outlier_side=' + str(outlier_side) + ')'
    elif method == 'zScore':
        keepTrials        = np.where(df_epos['zVar'] > zthreshold, 0, 1)
        title = 'ZVarience threshold of ' + str(zthreshold)
    elif method == 'none':
        title = 'no additional artefact rejection '
        keepTrials        = np.ones((1,ntrials))[0]

    df_epos['keepTrial'] = keepTrials
    df_keeps = df_epos[df_epos['keepTrial'] == 1]
    print(str(ntrials - len(df_keeps)) + ' trials discarded')

    # get the clean data
    keep_idx    = np.asarray(np.where(keepTrials),int)
    clean_dat    = np.squeeze(dat[keep_idx])

    #recalculate the var for chan
    clean_var    = np.var(clean_dat, 2)
    x_chans_c    = np.mean(clean_var,0)
    x_chans_c    = [x * 10**6 for x in x_chans_c]

    df_chans_c   = pd.DataFrame({'var': x_chans_c, 'chans': y_chans})


    # Plot everything
    fig, axis = plt.subplots(2, 2, figsize=(12, 12))
    axis[0,0].set_ylim([0, max(y_epos) + min(y_epos)*2])
    axis[0,1].set_xlim([0, max(x_chans)+ min(x_chans)*2])
    axis[1,0].set_ylim([0, max(df_keeps['var'])+ min(df_keeps['var'])*2])
    axis[1,1].set_xlim([0, max(x_chans_c)+ min(x_chans_c)*2])

    axis[0,0].set_title(title)
    sns.scatterplot(x = 'epochs', y = 'var', hue = 'keepTrial', hue_order = [1,0], ax = axis[0,0], data = df_epos)
    sns.scatterplot(x = 'var', y = 'chans', ax = axis[0,1], data = df_chans)
    sns.scatterplot(x = 'epochs', y = 'var', ax = axis[1,0], data =df_keeps)
    sns.scatterplot(x = 'var', y = 'chans', ax = axis[1,1], data = df_chans_c)



    return axis, keep_idx