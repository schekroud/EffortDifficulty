#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:15:01 2024

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

glmdir = op.join(wd, 'glms', 'stim1locked', 'erp', 'glm2')

times = np.load(op.join(glmdir, 'glm_timerange.npy'))
#save regressor names and contrast names in the order they are in, to help know what is what
regnames  = np.load(op.join(glmdir, 'regressor_names.npy'))
copenames = np.load(op.join(glmdir, 'contrast_names.npy'))


betas = dict();
copes = dict();

for iname in regnames:
    betas[iname] = []
for iname in copenames:
    copes[iname] = []

#read in all the data
for i in subs:
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    print(f'- - - - working on subject {i} - - - -')
    
    for iname in regnames:
        betas[iname].append(
            mne.read_evokeds(op.join(glmdir, f's{i}_stim1locked_{iname}_beta-ave.fif'))[0]
            )
        
    for iname in copenames:
        copes[iname].append(
            mne.read_evokeds(op.join(glmdir, f's{i}_stim1locked_{iname}_cope-ave.fif'))[0]
            )    

#%%
colors = dict(intercept = '#000000', prevtrlcorrectness = '#2166ac', trialnumber = '#8c510a', correct = '#1a9850', incorrect = '#b2182b')

gmean = mne.grand_average(betas['intercept'])
gmean.plot_joint(times = np.arange(0, 0.5, 0.05))

correctness = mne.grand_average(betas['prevtrlcorrectness'])
correctness.plot_joint(times = np.arange(0, 0.5, 0.05), 
                       ts_args = dict(ylim = dict(eeg = [-1, 1])),
                       topomap_args = dict(vlim = [-1, 1]))

mne.viz.plot_compare_evokeds(
    evokeds = dict(correct   = copes['correct'],
                   incorrect = copes['incorrect'],
                   diff      = copes['prevtrlcorrectness']),
    colors = dict(correct   = '#1a9850',
                  incorrect = '#b2182b',
                  diff      = '#2166ac'),
    picks = 'POz', ci = 0.68
    )



