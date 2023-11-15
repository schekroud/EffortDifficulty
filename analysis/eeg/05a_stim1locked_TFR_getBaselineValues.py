# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:53:35 2023

@author: sammirc
"""

import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
%matplotlib

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')

from funcs import getSubjectInfo, gesd, plot_AR

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)
import glmtools as glm


subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
#%%
for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    tfr = mne.time_frequency.read_tfrs(param['stim1locked'].replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]#.replace('stim1locked', 'stim1locked_cleaned').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
        
    #get all trials + frequencies, in the baseline period
    
    tfrdat = tfr.copy().crop(tmin = -2.7, tmax = -2.4).data #get just the array of the data
    blinevals = tfrdat.mean(axis = -1) # trials x channels x frequency
    
    np.save(op.join(wd, 'eeg', 's%02d'%i, 'EffortDifficulty_s%02d_stim1locked_baselineValues.npy'%i), blinevals) #save the baseline values per trial channel and frequency to file to apply later on
    
    tfrdat_alpha = tfr.copy().crop(tmin = -2.7, tmax = -2.4, fmin = 8, fmax = 12).data    
    bline_alpha = tfrdat_alpha.mean(axis=-1)
    np.save(op.join(wd, 'eeg', 's%02d'%i, 'EffortDifficulty_s%02d_stim1locked_baselineValues_AlphaOnly.npy'%i), bline_alpha) #save the baseline values per trial channel and frequency to file to apply later on
