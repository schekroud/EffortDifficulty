# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:27:50 2023

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


subs = np.array([10, 11, 12, 13, 14, 15, 16])

regs = ['grandmean', 'correctness', 'difficulty']
contrasts = ['grandmean', 'corrvsincorr', 'difficulty', 'correct', 'incorrect']

addtopath = '_baselined'
# addtopath = ''
b = dict()
c = dict()
t = dict()
for i in regs:
    b[i] = []
for i in contrasts:
    c[i] = []
    t[i] = []

for i in subs:
    print('\nworking on subject ' + str(i) +'\n')
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    for name in regs:
        b[name].append(
            mne.time_frequency.read_tfrs(  fname = op.join(wd, 'glms', 'stim1locked', 'glm1', 
                                          param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_beta-tfr.h5'))[0].crop(tmin=-2.85, tmax=2.85))
            
    for name in contrasts:
        c[name].append(mne.time_frequency.read_tfrs(fname = op.join(wd, 'glms', 'stim1locked', 'glm1', 
                                      param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_cope-tfr.h5'))[0].crop(tmin=-2.85, tmax=2.85))
        t[name].append(mne.time_frequency.read_tfrs(fname = op.join(wd, 'glms', 'stim1locked', 'glm1', 
                                      param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_tstat-tfr.h5'))[0].crop(tmin=-2.85, tmax=2.85))

#%%
alltimes = mne.grand_average(b['correctness']).times
allfreqs = mne.grand_average(b['correctness']).freqs
#%%
timefreqs       = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4),
                   (.4, 22):(.4, 16),
                   (.6, 22):(.4, 16),
                   (.8, 22):(.4, 16)}
timefreqs_alpha = {(-.5, 10):(.6, 4),
                   (-.3, 10):(.6, 4),
                   (.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4),
                   (1., 10):(.4, 4),
                   (1.2, 10):(.4, 4)}
visleftchans  = ['PO3', 'PO7', 'O1']
visrightchans = ['PO4','PO8','O2']
motrightchans = ['C4']  #channels contra to the left hand (space bar)
motleftchans  = ['C3']   #channels contra to the right hand (mouse)
topoargs = dict(outlines= 'head', contours = 0)
topoargs_t = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2)
plt.close('all')
posteriorchannels = ['PO7', 'PO3', 'O1', 'PO8', 'PO4', 'O2']
#%%

#make jointplots showing the regressors entered into the model

for beta in regs:
    if beta == 'grandmean':
        baseline = (-2.7, -2.4)
        baseline = None
    else:
        baseline = None
        # baseline = (-2.7, -2.4)
    dat2plot = mne.grand_average(b[beta])
    dat2plot.plot_joint(baseline = baseline, timefreqs = timefreqs_alpha,
                        topomap_args = topoargs,
                        #image_args = dict(picks = posteriorchannels),
                        title = 'beta: '+beta)
#%%

#make jointplots showing the copes calculated by the model
for contrast in contrasts:
    if contrast in ['grandmean', 'correct', 'incorrect']:
        baseline = (-2.7, -2.4)
        baseline = None
    else:
        baseline = None
    dat2plot = mne.grand_average(c[contrast])
    dat2plot.plot_joint(baseline = baseline, timefreqs = timefreqs_alpha, topomap_args = topoargs, title = 'cope: '+contrast)


# mne.grand_average(c['corrvsincorr']).plot(picks = posteriorchannels, combine = 'mean')
#%%

#make jointplots showing the mean t-stat for copes calculated by the model
for contrast in contrasts:
    if contrast in ['grandmean', 'correct', 'incorrect']:
        baseline = (-2.7, -2.4)
        baseline = None
    else:
        baseline = None
    dat2plot = mne.grand_average(t[contrast])
    dat2plot.plot_joint(baseline = baseline, timefreqs = timefreqs_alpha, topomap_args = topoargs, title = 'tstat for cope: '+contrast)

