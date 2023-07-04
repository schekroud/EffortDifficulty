#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:51:01 2023

@author: sammi
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

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')

from funcs import getSubjectInfo, gesd, plot_AR

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)

subs = np.array([10, 11, 12, 13, 14, 15, 16])

correct = []
incorrect = []
for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    
    epochs = mne.read_epochs(fname = param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned'), preload = True)
    epochs.apply_baseline((-.15, 0)) #baseline 250ms prior to feedback
    epochs.resample(500) #resample to 500Hz
    
    iCorrect = epochs['PerceptDecCorrect == 1']
    iIncorrect = epochs['PerceptDecCorrect == 0']
    
    correct.append(iCorrect.average())
    incorrect.append(iIncorrect.average())

#%%
topoargs = dict(contours=0)
#get grand averages

gave_corr = mne.grand_average(correct)
gave_incorr = mne.grand_average(incorrect)
gave_diff = mne.combine_evoked([gave_corr, gave_incorr], weights = [1, -1])

gave_diff.plot_joint(times = np.arange(0, 0.6, 0.1), topomap_args = topoargs, title = 'diff')
gave_corr.plot_joint(times = np.arange(0, 0.6, 0.1), topomap_args = topoargs, title = 'correct')
gave_incorr.plot_joint(times = np.arange(0, 0.6, 0.1), topomap_args = topoargs, title = 'incorrect')