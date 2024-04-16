#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:44:52 2024

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

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

import progressbar
progressbar.streams.flush()

accuracies = np.zeros(shape = [subs.size, 1251])

subcount = -1
for i in subs:
    subcount += 1
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    print(f'- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'), preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -1, tmax = 1.5) #crop for speed and relevance
    
    #drop some trials
    epochs = epochs['fbtrig != 62'] #drop timeout trials
    epochs = epochs['diffseqpos > 3'] #remove the first three trials of each new difficulty sequence
    #so we only look at trials where they should have realised the difficulty level
    epochs = epochs['prevtrlfb in ["correct", "incorrect"]'] #drop trials where previous trial was a timeout, as this is a big contributor to some effects
    
    if i == 16:
        epochs = epochs['blocknumber != 4']
    if i in [21, 25]:
        #these participants had problems in block 1
        epochs = epochs['blocknumber > 1']
        
    data   = epochs._data.copy() #get the data matrix
    bdata  = epochs.metadata.copy() 
    
    [ntrials, nfeatures, ntimes] = data.shape #get data shape
    
    #we want to predict the orientation of the stimulus relative to the vertical cardinal axis.
    #this is coded in arraytiltcond
    #0 = negative, 1 = positive (i.e. anticlockwise or clockwise relative to vertical)
    labels = bdata.arraytiltcond.to_numpy()
    
    
    # label_pred = np.zeros_like(labels)     * np.nan
    accuracy   = np.zeros(ntimes) * np.nan
    
    prediction = np.zeros(shape = [ntrials, ntimes]) * np.nan
    label_pred = np.zeros_like(prediction) * np.nan
    
    progressbar.streams.wrap_stderr()
    bar = progressbar.ProgressBar(max_value = ntimes).start()
    for tp in range(ntimes):
        bar.update(tp+1)
        X = data[:,:,tp].copy()
    
        pca = skl.decomposition.PCA(n_components = 0.95) #run pca decomposing to explain 95% of variance
        
        X_pca = pca.fit(X).transform(X)
        
        #train test set
        # cv = skl.model_selection.LeaveOneOut()
        cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 420)
        
        for train_index, test_index in cv.split(X, labels):
            x_train, x_test = X[train_index], X[test_index]
            y_train  = labels[train_index]
            
            #if using leave one out splits, need to reshape so scaler works
            # x_test = x_test.reshape(1, -1)
            
            scaler = skl.preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test  = scaler.transform(x_test)
    
            # clf = skl.linear_model.LogisticRegression(random_state = 420, solver = 'lbfgs', multi_class = 'ovr')
            clf = skl.linear_model.RidgeClassifier(fit_intercept = True)
            
            
            
            clf.fit(x_train, y_train)
            
            label_pred[test_index,tp] = clf.predict(x_test)
            # prediction[test_index,tp] = np.squeeze(clf.predict_proba(x_test))[clf.predict(x_test)]
            # score[test_index, tp] = clf.score(x_train, y_train)
        
        accuracy[tp] = skl.metrics.accuracy_score(labels,label_pred[:,tp])
    accuracies[subcount] = accuracy
            
#%%
accs = sp.ndimage.gaussian_filter1d(accuracies.copy(), axis = 1, sigma = 10)

gave_acc = np.mean(accs, axis = 0) #average across subjects
sem_acc  = sp.stats.sem(accs, axis = 0, ddof = 0)

fig = plt.figure()
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
            
            
            
            
    