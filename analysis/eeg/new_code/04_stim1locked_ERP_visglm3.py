#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:20:31 2024

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

glmdir = op.join(wd, 'glms', 'stim1locked', 'erp', 'glm3b')

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
colors = dict(intercept = '#000000',diff2 = '#e41a1c', diff4 = '#fc8d62', diff8 = '#41ab5d', diff12 = '#006d2c', trialnumber = '#8c510a')
colors = dict(intercept = '#000000', difficulty = '#3182bd')

difficulty = mne.grand_average(betas['difficulty'])
difficulty.plot_joint(times = np.arange(0, 0.5, 0.05), 
                       ts_args = dict(ylim = dict(eeg = [-1, 1])),
                       topomap_args = dict(vlim = [-1, 1]))

# mne.viz.plot_compare_evokeds(
#     evokeds = dict(
#         diff2 = betas['diff2'],
#         diff4 = betas['diff4'],
#         diff8 = betas['diff8'],
#         diff12 = betas['diff12']
#         ),
#     colors = dict(
#         diff2 = '#e41a1c',
#         diff4 = '#fc8d62',
#         diff8 = '#41ab5d',
#         diff12 = '#006d2c'
#         ),
#     picks = 'POz', ci = 0.68
#     )


mne.viz.plot_compare_evokeds(
    evokeds = dict(
        intercept  = betas['intercept'],
        difficulty = betas['difficulty']
        ),
    colors = dict(
        intercept = '#000000',
        difficulty = '#3182bd'
        ),
    picks = 'FCz', ci = 0.68
    )
