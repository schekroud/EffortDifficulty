# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:05:03 2024

@author: sammirc
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
mne.viz.set_browser_backend('qt')
import glmtools as glm

loc = 'workstation'
if loc == 'laptop':
    sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
elif loc == 'workstation':
    sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

from funcs import getSubjectInfo
from decoding_functions import run_decoding
from joblib import parallel_backend

os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

figpath = op.join(wd, 'figures', 'decoding', 'LDA')

import progressbar
progressbar.streams.flush()
accuracies = np.zeros(shape = [subs.size, 1001])
diffaccuracies = np.zeros(shape = [subs.size, 4, 1001])
use_vischans_only  = True
use_pca            = False #using pca after selecting only visual channels is really bad for the decoder apparently
smooth_singletrial = True
testtype           = 'rskf'

progressbar.streams.wrap_stderr()
bar = progressbar.ProgressBar(max_value = len(subs)).start()
subcount = -1
for i in subs:
    subcount += 1
    bar.update(subcount+1)
    # sub   = dict(loc = 'laptop', id = i)
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    print(f'\n- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'),
                                verbose = 'ERROR', preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -0.5, tmax = 1.5) #crop for speed and relevance  
    #drop some trials
    # epochs = epochs['fbtrig != 62'] #drop timeout trials
    # epochs = epochs['diffseqpos > 3'] #remove the first three trials of each new difficulty sequence
    # #so we only look at trials where they should have realised the difficulty level
    # epochs = epochs['prevtrlfb in ["correct", "incorrect"]'] #drop trials where previous trial was a timeout, as this is a big contributor to some effects
    
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
    if smooth_singletrial:
        data = sp.ndimage.gaussian_filter1d(data.copy(), sigma = 7) #this is equivalent to smoothing with a 14ms gaussian blur as 500Hz sample rate.
        #this should help with some denoising
    bdata  = epochs.metadata.copy() 
    [ntrials, nfeatures, ntimes] = data.shape #get data shape -- [trials x channels x timepoints]

    #stimulus orientations are drawn from between -60 - -30, and 30-60
    #these orientations are drawn randomly from a uniform distribution, so by design there is no bias
    #lets just split these immediately in two, and do a four class decoding
    # -30>-45, -46>-60, 30-45, 46-60
    stim1oris = bdata.stim1ori.to_numpy()
    labels = np.zeros_like(stim1oris) 
    labels[np.isin(stim1oris, np.arange(-60, -45))] = 1 #between -60 and -46 
    labels[np.isin(stim1oris, np.arange(-45, -29))] = 2 #between -45 and -30
    labels[np.isin(stim1oris, np.arange(30, 46))]   = 3 #between 30 and 45 (inclusive)
    labels[np.isin(stim1oris, np.arange(46, 61))]   = 4 #between 46 and 60 

    
    accuracy   = np.zeros(ntimes) * np.nan
    difficultyOris = bdata.difficultyOri.copy()
    diffseqpos = bdata.diffseqpos.copy()
    prediction = np.zeros(shape = [ntrials, ntimes]) * np.nan
    label_pred = np.zeros_like(prediction) * np.nan
    diffaccs = np.zeros(shape = [4, ntimes]) * np.nan #store label predictions here
    
    # progressbar.streams.wrap_stderr()
    # bar = progressbar.ProgressBar(max_value = ntimes).start()
    with parallel_backend('loky', n_jobs = 4):
        for tp in range(ntimes):
            # bar.update(tp+1)
            preds = run_decoding(data, labels, tp, use_pca, testtype, classifier = 'LDA', nsplits = 10)
            label_pred[:,tp] = preds
            accuracy[tp] = skl.metrics.accuracy_score(labels,label_pred[:,tp])
            
    diff2  = label_pred[np.logical_and(difficultyOris.eq(2),  diffseqpos.ge(4))]
    diff4  = label_pred[np.logical_and(difficultyOris.eq(4),  diffseqpos.ge(4))]
    diff8  = label_pred[np.logical_and(difficultyOris.eq(8),  diffseqpos.ge(4))]
    diff12 = label_pred[np.logical_and(difficultyOris.eq(12), diffseqpos.ge(4))]
    
    for tp in range(ntimes):
        diffaccs[0, tp] = skl.metrics.accuracy_score(labels[np.logical_and(difficultyOris.eq(2),  diffseqpos.ge(4))],  diff2[:, tp])
        diffaccs[1, tp] = skl.metrics.accuracy_score(labels[np.logical_and(difficultyOris.eq(4),  diffseqpos.ge(4))],  diff4[:, tp])
        diffaccs[2, tp] = skl.metrics.accuracy_score(labels[np.logical_and(difficultyOris.eq(8),  diffseqpos.ge(4))],  diff8[:, tp])
        diffaccs[3, tp] = skl.metrics.accuracy_score(labels[np.logical_and(difficultyOris.eq(12), diffseqpos.ge(4))], diff12[:, tp])
    
    accuracies[subcount] = accuracy
    diffaccuracies[subcount] = diffaccs

#%%
# accs = sp.ndimage.gaussian_filter1d(accuracies.copy(), axis = 1, sigma = 10)
accs = accuracies.copy()

gave_acc = np.mean(accs, axis = 0) #average across subjects
sem_acc  = sp.stats.sem(accs, axis = 0, ddof = 0)

fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.plot(epochs.times, gave_acc, color = '#3182bd') #smooth with gaussian blur of 5 samples (10ms)
ax.fill_between(epochs.times,
                np.add(gave_acc, sem_acc),
                np.subtract(gave_acc, sem_acc),
                color = '#3182bd', alpha = 0.3, edgecolor = None)
ax.axvline(x = 0, ls = 'dashed', color = 'k')
ax.axhline(y = 0.25, ls = 'dashed', color = 'k')
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('classifier accuracy');
ax.set_ylim([0.24, 0.32])
fig.savefig(op.join(figpath, f'decoding_LDAClassifier_allTrials_vischansonly{use_vischans_only}_usePca{use_pca}_{testtype}.pdf'), format = 'pdf', dpi = 400)

#%%
diffssmoothed = sp.ndimage.gaussian_filter1d(diffaccuracies.copy(), axis = 2, sigma = 2)

diffs_mean = np.mean(diffssmoothed, axis = 0)
diffs_sem  = sp.stats.sem(diffssmoothed, axis = 0, ddof = 0)

labels, colors = ['diff2', 'diff4', 'diff8', 'diff12'], ['#e41a1c', '#fc8d62', '#41ab5d', '#006d2c']

fig = plt.figure(figsize = [6, 4])
ax = fig.add_subplot(111)
for idiff in range(4):
    ax.plot(epochs.times, diffs_mean[idiff], label = labels[idiff], color = colors[idiff], lw = 1)
    ax.fill_between(epochs.times,
                    np.add(diffs_mean[idiff], diffs_sem[idiff]),
                    np.subtract(diffs_mean[idiff], diffs_sem[idiff]),
                    color = colors[idiff], alpha = 0.3, edgecolor = None)
ax.axvline(x = 0, ls = 'dashed', color = 'k')
ax.axhline(y = 0.25, ls = 'dashed', color = 'k')
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('classifier accuracy');
ax.set_ylim([0.2, 0.32])
ax.set_xlim([-0.2, 1])
ax.legend(loc = 'upper left')
    
    
    
    
    
    
    