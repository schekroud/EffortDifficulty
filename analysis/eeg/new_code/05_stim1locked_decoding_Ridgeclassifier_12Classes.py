#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:30:44 2024

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

os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

figpath = op.join(wd, 'figures', 'decoding', '12class')

import progressbar
progressbar.streams.flush()

nsamples = 200 #samples in the epoch at 100hz sample rate
# nsamples = 1001 #samples in the epoch at 500hz sample rate
accuracies = np.zeros(shape = [subs.size, nsamples])
diffaccuracies = np.zeros(shape = [subs.size, 4, nsamples])
difficulty_evidences = np.zeros_like(diffaccuracies) * np.nan
evidence   = np.zeros_like(accuracies)
use_vischans_only  = True
use_pca            = False #using pca after selecting only visual channels is really bad for the decoder apparently
smooth_singletrial = True
testtype           = 'rskf'
classifier         = 'LDA' #LDA and ridge are the two options that work at the moment

nbins = 6
centprobs = np.zeros(shape = [subs.size, nbins-1, nsamples]) #store the average tuning curve for each participant

progressbar.streams.wrap_stderr()
bar = progressbar.ProgressBar(max_value = len(subs)).start()
subcount = -1
for i in subs:
    subcount += 1
    bar.update(subcount+1)
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
        # data = sp.ndimage.gaussian_filter1d(data.copy(), sigma = 7) #this is equivalent to smoothing with a 14ms gaussian blur as 500Hz sample rate.
        data = sp.ndimage.gaussian_filter1d(data.copy(), sigma = 2) #this is equivalent to smoothing with a 20ms gaussian blur as 100Hz sample rate.
        #this should help with some denoising
        
    bdata  = epochs.metadata.copy() 
    [ntrials, nfeatures, ntimes] = data.shape #get data shape -- [trials x channels x timepoints]

    orientations = bdata.stim1ori.to_numpy()
    
    posbins = np.linspace(30, 60+1, nbins)
    negbins = np.linspace(-60, -30+1, nbins)
    labelbinspos = np.digitize(orientations, posbins)
    labelbinsneg = np.digitize(orientations, negbins)
    labelbinsneg = np.where(labelbinsneg == nbins, 0, labelbinsneg)
    labelbinspos = np.where(labelbinspos == 0, 0, labelbinspos+(nbins-1))
    
    labelbins = np.where(labelbinsneg == 0, labelbinspos, labelbinsneg)
    #pd.value_counts(labelbins)
    nclasses = np.unique(labelbins).size
        
    
    #pre-create some variables to store info
    label_pred = np.zeros(shape = [ntrials, ntimes]) * np.nan #store the predicted label for each trial and time
    accuracy = np.zeros(ntimes) #store across-trial classifier accuracy at each time point
    
    predictedprobs     = np.zeros(shape = [ntrials, nclasses, ntimes]) #store the P(class) for each class, per trial and time point
    # centred_probabilities = np.zeros_like(predictedprobs) #store P(class), centred on the predicted class
    
    for tp in range(ntimes): #for each time point
        preds, predprobas = run_decoding_predproba(data, labelbins, tp, use_pca, testtype, classifier = classifier) #run decoding
        #this outputs:
        # preds -- the predicted class for each trial (ntrials)
        # predprobas - the P(class) for each class (ntrials x nclasses)
        
        label_pred[:,tp]        = preds.astype(int) #store the across-trial predicted class labels for this time point
        predictedprobs[:,:,tp]  = predprobas         #store the class predicted probabilities
    
    # for each time point, we want to get the accuracy across trials. We also went to get the single-trial evidence for the stimulus
    # the predicted probability of the true label)
    for tp in range(ntimes):
        accuracy[tp] = skl.metrics.accuracy_score(labelbins, label_pred[:, tp])
    accuracies[subcount] = accuracy #store the within subject, across trial, classifier accuracy
    
    singtrlev = np.zeros_like(label_pred) * np.nan
    #here, we want to get the predicted probability of the true class, at each time point per trial
    for trl in range(ntrials):
        itrl = predictedprobs[trl] #get P(class) for each class on this trial, across all time points
        ilabel = labelbins[trl]    #get the true class on this trial
        truetimecourse = itrl[ilabel-1] #get the across time probability for the true stimulus class
        singtrlev[trl] = truetimecourse
    evidence[subcount] = np.mean(singtrlev, axis=0)

    #get accuracies separately for each difficulty level
    d2 = bdata.difficultyOri.eq(2)
    d4 = bdata.difficultyOri.eq(4)
    d8 = bdata.difficultyOri.eq(8)
    d12 = bdata.difficultyOri.eq(12)
    
    for tp in range(ntimes):
        diffaccuracies[subcount, 0, tp] = skl.metrics.accuracy_score(labelbins[d2],  label_pred[labelbins[d2],  tp])
        diffaccuracies[subcount, 1, tp] = skl.metrics.accuracy_score(labelbins[d4],  label_pred[labelbins[d4],  tp])
        diffaccuracies[subcount, 2, tp] = skl.metrics.accuracy_score(labelbins[d8],  label_pred[labelbins[d8],  tp])
        diffaccuracies[subcount, 3, tp] = skl.metrics.accuracy_score(labelbins[d12], label_pred[labelbins[d12], tp])
        
    difficulty_evidences[subcount,0] = singtrlev[d2].mean(0)
    difficulty_evidences[subcount,1] = singtrlev[d4].mean(0)
    difficulty_evidences[subcount,2] = singtrlev[d8].mean(0)
    difficulty_evidences[subcount,3] = singtrlev[d12].mean(0)


plottimes = epochs.times

#%%
#plot across-subject accuracy
accs = sp.ndimage.gaussian_filter1d(accuracies, sigma = 3)
# accs = accuracies.copy()

gmean_acc = accs.mean(axis=0)
sem_acc   = sp.stats.sem(accs, axis = 0, ddof = 0)


fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1)
ax.axhline(1/nclasses, ls = 'dashed', color = '#000000', lw = 1)
ax.plot(plottimes, gmean_acc, color = '#3182bd', lw = 2)
ax.fill_between(plottimes,
                np.add(gmean_acc, sem_acc), np.subtract(gmean_acc, sem_acc),
                alpha = 0.3, color = '#3182bd', edgecolor = None)
ax.set_ylabel('classifier accuracy')
ax.set_xlabel('time relative to stimulus 1 onset (s)')
fig.suptitle(f'{classifier}')
# ax.set_ylim([0.065, 0.1])
# fig.savefig(op.join(figpath, f'{classifier}_12classdecoding_gmeanAccuracy_usevischansonly{use_vischans_only}_pca{use_pca}.pdf'), dpi = 300)


evs = sp.ndimage.gaussian_filter1d(evidence, sigma=3)
# evs = evidence.copy()

gmean_ev = evs.mean(axis=0)
sem_ev   = sp.stats.sem(evs, ddof = 0, axis = 0)


fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1)
# ax.axhline(1/nclasses, ls = 'dashed', color = '#000000', lw = 1)
ax.plot(plottimes, gmean_ev, color = '#3182bd', lw = 2)
ax.fill_between(plottimes,
                np.add(gmean_ev, sem_ev), np.subtract(gmean_ev, sem_ev),
                alpha = 0.3, color = '#3182bd', edgecolor = None)
ax.set_ylabel('probability of true stimulus class')
ax.set_xlabel('time relative to stimulus 1 onset (s)')
fig.suptitle(f'{classifier}')
# ax.set_ylim([0.065, 0.09])
# fig.savefig(op.join(figpath, f'{classifier}_12classdecoding_gmeanSingleTrlEvidence_usevischansonly{use_vischans_only}_pca{use_pca}.pdf'), dpi = 300)


#%% plot by difficulty level

diffaccs = sp.ndimage.gaussian_filter1d(diffaccuracies, sigma=3)
# diffaccs = diffaccuracies.copy()

gmean_diffacc = diffaccs.mean(axis=0)
sem_diffacc   = sp.stats.sem(diffaccs, axis = 0, ddof = 0)
labels, colors = ['diff2', 'diff4', 'diff8', 'diff12'], ['#e41a1c', '#fc8d62', '#41ab5d', '#006d2c']


fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1)
ax.axhline(1/nclasses, ls = 'dashed', color = '#000000', lw = 1)
for idiff in range(len(labels)):
    ax.plot(plottimes, gmean_diffacc[idiff], lw = 1, label = labels[idiff], color = colors[idiff])
    ax.fill_between(plottimes,
                    np.add(gmean_diffacc[idiff], sem_diffacc[idiff]), np.subtract(gmean_diffacc[idiff], sem_diffacc[idiff]),
                    alpha = 0.3, color = colors[idiff], edgecolor = None)
ax.set_ylabel('classifier accuracy')
ax.set_xlabel('time relative to stimulus 1 onset (s)')
ax.legend(loc = 'lower left')
fig.suptitle(f'{classifier}')
# ax.set_ylim([0.065, 0.1])
ax.set_xlim([-0.5, 1])
# fig.savefig(op.join(figpath, f'{classifier}_12classdecoding_gmeanAccuracy_usevischansonly{use_vischans_only}_pca{use_pca}.pdf'), dpi = 300)

#%%

diffaccs = sp.ndimage.gaussian_filter1d(difficulty_evidences, sigma=2)
# diffaccs = difficulty_evidences.copy()

gmean_diffacc = diffaccs.mean(axis=0)
sem_diffacc   = sp.stats.sem(diffaccs, axis = 0, ddof = 0)
labels, colors = ['diff2', 'diff4', 'diff8', 'diff12'], ['#e41a1c', '#fc8d62', '#41ab5d', '#006d2c']


fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
ax.axvline(0, ls = 'dashed', color = '#000000', lw = 1)
# ax.axhline(1/nclasses, ls = 'dashed', color = '#000000', lw = 1)
for idiff in range(len(labels)):
    ax.plot(plottimes, gmean_diffacc[idiff], lw = 1, label = labels[idiff], color = colors[idiff])
    ax.fill_between(plottimes,
                    np.add(gmean_diffacc[idiff], sem_diffacc[idiff]), np.subtract(gmean_diffacc[idiff], sem_diffacc[idiff]),
                    alpha = 0.3, color = colors[idiff], edgecolor = None)
ax.set_ylabel('probability of true stimulus class')
ax.set_xlabel('time relative to stimulus 1 onset (s)')
ax.set_xlim([-0.2, 1])
ax.legend(loc = 'lower left', ncols=4, frameon=False)
fig.suptitle(f'{classifier}')








