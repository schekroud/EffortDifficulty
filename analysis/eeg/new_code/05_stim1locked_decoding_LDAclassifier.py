#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:55:15 2024

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

os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

figpath = op.join(wd, 'figures', 'decoding', 'LDA')


import progressbar
progressbar.streams.flush()

# accuracies = np.zeros(shape = [subs.size, 1251])
accuracies = np.zeros(shape = [subs.size, 1001])


use_vischans_only  = True
use_pca            = True #using pca after selecting only visual channels is really bad for the decoder apparently
smooth_singletrial = True

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
                            'O1', 'Oz', 'O2'
            ])
    
    data   = epochs._data.copy() #get the data matrix
    if smooth_singletrial:
        data = sp.ndimage.gaussian_filter1d(data.copy(), sigma = 8) #this is equivalent to smoothing with a 14ms gaussian blur as 500Hz sample rate.
        #this should help with some denoising
    bdata  = epochs.metadata.copy() 
    [ntrials, nfeatures, ntimes] = data.shape #get data shape -- [trials x channels x timepoints]
    
    #we want to predict the orientation of the stimulus relative to the vertical cardinal axis.
    #this is coded in arraytiltcond
    #0 = negative, 1 = positive (i.e. anticlockwise or clockwise relative to vertical)
    labels = bdata.arraytiltcond.to_numpy()
    accuracy   = np.zeros(ntimes) * np.nan
    prediction = np.zeros(shape = [ntrials, ntimes]) * np.nan
    label_pred = np.zeros_like(prediction) * np.nan
    # prediction_probs = np.zeros(shape = )
    for tp in range(ntimes):
        X = data[:,:,tp].copy()
        pca = skl.decomposition.PCA(n_components = 0.95) #run pca decomposing to explain 95% of variance
        X_pca = pca.fit(X).transform(X)
        
        #train test set
        testtype = 'rskf'
        if testtype == 'leaveoneout':
            cv = skl.model_selection.LeaveOneOut()
        elif testtype == 'rskf':
            cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = 10, n_repeats = 1, random_state = 420)
        
        for train_index, test_index in cv.split(X, labels):
            if use_pca:
                x_train, x_test = X_pca[train_index], X_pca[test_index]
            else:
                x_train, x_test = X[train_index], X[test_index]
            y_train  = labels[train_index]
            
            #if using leave one out splits, need to reshape so scaler works
            if testtype == 'leaveoneout':
                x_test = x_test.reshape(1, -1)
            
            scaler = skl.preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test  = scaler.transform(x_test)

            clf = skl.discriminant_analysis.LinearDiscriminantAnalysis(solver = 'svd')
            clf.fit(x_train, y_train)
            label_pred[test_index,tp] = clf.predict(x_test) #get binary class predictions
            
            #you can get single trial evidence (probability of correct class assignment) if you want to so single trial analyses
            # predprobas = clf.predict_proba(x_test) #this gives probability assigned to either 0 or 1 (it is [ntrials x nClasses] in shape)
            # #we want to get the probability of assigning the *true* label
            # truelabels = labels[test_index]
            # classprobs = np.array([predprobas[i][truelabels[i]] for i in range(len(predprobas))])
            
            
            # prediction[test_index,tp] = np.squeeze(clf.predict_proba(x_test))[clf.predict(x_test)]
            # score[test_index, tp] = clf.score(x_train, y_train)
        
        accuracy[tp] = skl.metrics.accuracy_score(labels,label_pred[:,tp])
    accuracies[subcount] = accuracy

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
ax.axhline(y = 0.5, ls = 'dashed', color = 'k')
ax.set_xlabel('time relative to stim1 onset (s)')
ax.set_ylabel('classifier accuracy')
ax.set_ylim([0.48, 0.6])
# ax.set_xlim([-0.5, 1])
fig.savefig(op.join(figpath, f'decoding_LDA_allTrials_vischansonly{use_vischans_only}_usePca{use_pca}_{testtype}.pdf'), format = 'pdf', dpi = 400)
            