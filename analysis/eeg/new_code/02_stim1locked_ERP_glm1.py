#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:00:47 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import sklearn as skl
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

glmdir = op.join(wd, 'glms', 'stim1locked', 'glm1')

for i in subs:
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    print(f'- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'), preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -1, tmax = 1.5) #crop for speed and relevance
    
    data   = epochs._data.copy() #get the data matrix
    
    #build model
    acc      = epochs.metadata.PerceptDecCorrect.to_numpy()
    acc      = np.where(acc == 1, 1, -1)
    acc      = sp.stats.zscore(acc) #zscore this across trials

    DC = glm.design.DesignConfig()
    DC.add_regressor(name = 'intercept', rtype = 'Constant') #add intercept to model average lateralisation
    DC.add_regressor(name = 'correctness', rtype = 'Parametric', datainfo = 'correctness', preproc = None)
    DC.add_simple_contrasts() #add basic diagonal matrix for copes
    DC.add_contrast(values = [1, 1], name = 'correct')
    DC.add_contrast(values = [1,-1], name = 'incorrect')

    #create glmdata object
    glmdata = glm.data.TrialGLMData(data = data, time_dim = 2, sample_rate = 500,
                                    #add in metadata that's used to construct the design matrix
                                    correctness = acc)
    glmdes = DC.design_from_datainfo(glmdata.info)
    
    print('\n - - - - -  running glm - - - - - \n')
    model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model 
    betas = model.betas.copy()
    copes = model.copes.copy()
    tstats = model.tstats.copy()
    
    regnames  = model.regressor_names
    copenames = model.contrast_names
    ntrls = model.num_observations
    
    tmin = epochs.tmin
    info = epochs.info
    times = epochs.times
    
    
    
    for ireg in range(len(regnames)):
        ibeta = betas[ireg].copy()
        tl = mne.EvokedArray(info = info, nave = ntrls, tmin = tmin, data = ibeta)
        tl.save(fname = op.join(glmdir, f's{i}_stim1locked_{regnames[ireg]}_beta-ave.fif'), overwrite = True)
        
    for icope in range(len(copenames)):
        icontr = copes[icope].copy()
        tl = mne.EvokedArray(info = info, nave = ntrls, tmin = tmin, data = icontr)
        tl.save(fname = op.join(glmdir, f's{i}_stim1locked_{copenames[icope]}_cope-ave.fif'), overwrite = True)


    if i == 10: #for first subject, lets also save a couple things for this glm to help with visualising stuff
        #going to save the times
        np.save(file = op.join(glmdir, 'glm_timerange.npy'), arr= times)
        #save regressor names and contrast names in the order they are in, to help know what is what
        np.save(file = op.join(glmdir, 'regressor_names.npy'), arr = model.regressor_names)
        np.save(file = op.join(glmdir, 'contrast_names.npy'),  arr = model.contrast_names)
    
    del(ibeta)
    del(icope)
    del(tl)
    
    
    
    
    
    