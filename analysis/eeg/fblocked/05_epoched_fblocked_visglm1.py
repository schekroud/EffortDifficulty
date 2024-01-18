# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:55:34 2023

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

sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gesd, plot_AR

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
glmdir = op.join(wd, 'glms', 'fblocked', 'glm1')

betas = dict()
copes = dict()
tstats = dict()

contrasts = np.load(op.join(glmdir, 'contrast_names.npy'))
betanames = np.load(op.join(glmdir, 'regressor_names.npy'))

for beta in betanames:
    betas[beta] = []
    for i in subs:
        sub   = dict(loc = 'workstation', id = i)
        param = getSubjectInfo(sub)
        betas[beta].append(mne.read_evokeds(fname=op.join(glmdir, param['subid']+'_fblockedEpoched_'+beta +'_betas-ave.fif' ))[0])

for cope in contrasts:
    copes[cope] = []
    tstats[cope] = []
    for i in subs:
        sub   = dict(loc = 'workstation', id = i)
        param = getSubjectInfo(sub)
        copes[cope].append(mne.read_evokeds(fname=op.join(glmdir, param['subid']+'_fblockedEpoched_'+cope +'_copes-ave.fif' ))[0])
        tstats[cope].append(mne.read_evokeds(fname=op.join(glmdir, param['subid']+'_fblockedEpoched_'+cope +'_tstats-ave.fif' ))[0])

#%%
correct = mne.grand_average(betas['correct']); times = correct.times;
frn       = mne.grand_average(copes['incorrvscorr'])
#%%


for channel in ['Fz', 'FCz', 'Cz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    correct    = betas['correct'],
                    incorrect  = betas['incorrect']),
                    # difference = betas['incorrvscorr']),
            colors = dict(
                    correct = '#2ca25f',
                    incorrect = '#e34a33'),
                    # difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'electrode '+channel, truncate_xaxis = False)

#%%

for channel in ['Fz', 'FCz', 'Cz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    correct  = copes['correct'],
                    incorrect  = copes['incorrect'],
                    difference = copes['incorrvscorr']),
            colors = dict(
                    correct = '#2ca25f',
                    incorrect = '#e34a33',
                    difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'electrode '+channel, truncate_xaxis = False)
    plt.xlim([-0.3, 1])
#%%

frn.plot_joint(times=np.arange(0, 0.6, 0.05), topomap_args = dict(contours=0))



