# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:20:22 2023

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
glmdir = op.join(wd, 'glms', 'fblocked', 'glm2')

betas = dict()
copes = dict()
tstats = dict()

contrasts = np.load(op.join(glmdir, 'contrast_names.npy'))
betanames = np.load(op.join(glmdir, 'regressor_names.npy'))
difficulties = [2, 4, 8, 12]

for difficulty in difficulties:
    betas['difficulty'+str(difficulty)] = dict()
    for beta in betanames:
        betas['difficulty'+str(difficulty)][beta] = []
        for i in subs:
            sub   = dict(loc = 'workstation', id = i)
            param = getSubjectInfo(sub)
            betas['difficulty'+str(difficulty)][beta].append(
                mne.read_evokeds(
                    fname=op.join(glmdir, param['subid']+'_fblockedEpoched_' + beta +'_' + 'difficulty' + str(difficulty) + '_betas-ave.fif'))[0].crop(tmin = -0.3, tmax = 0.7))

for difficulty in difficulties:
    copes['difficulty'+str(difficulty)] = dict()
    tstats['difficulty'+str(difficulty)] = dict()

    for cope in contrasts:
        copes['difficulty'+str(difficulty)][cope] = []
        tstats['difficulty'+str(difficulty)][cope] = []
        for i in subs:
            sub   = dict(loc = 'workstation', id = i)
            param = getSubjectInfo(sub)
            copes['difficulty'+str(difficulty)][cope].append(
                mne.read_evokeds(
                    fname=op.join(glmdir, param['subid']+'_fblockedEpoched_' + cope +'_' + 'difficulty' + str(difficulty) + '_copes-ave.fif'))[0].crop(tmin = -0.3, tmax = 0.7))
            tstats['difficulty'+str(difficulty)][cope].append(
                mne.read_evokeds(
                    fname=op.join(glmdir, param['subid']+'_fblockedEpoched_' + cope +'_' + 'difficulty' + str(difficulty) + '_tstats-ave.fif'))[0].crop(tmin = -0.3, tmax = 0.7))
#%%
correct = mne.grand_average(betas['difficulty2']['correct']); times = correct.times;
#%%

for difficulty in difficulties:
    fig = plt.figure(figsize = [6,6])
    ax = fig.add_subplot(111)
    for channel in ['FCz']:
        mne.viz.plot_compare_evokeds(
                evokeds = dict(
                        correct  = copes['difficulty'+str(difficulty)]['correct'],
                        incorrect  = copes['difficulty'+str(difficulty)]['incorrect'],
                        difference = copes['difficulty'+str(difficulty)]['incorrvscorr']),
                colors = dict(
                        correct = '#2ca25f',
                        incorrect = '#e34a33',
                        difference = '#000000'),
                legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
                show_sensors = False, title = 'difficulty %s, electrode %s'%(difficulty, channel), truncate_xaxis = False, axes=ax)
        # plt.xlim([-0.3, 1])

#%%
chan2use = ['FCz']
fig = plt.figure(figsize = [8,4])
ax = fig.add_subplot(111)
for channel in chan2use:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                diff2  = copes['difficulty2']['incorrect'],
                diff4  = copes['difficulty4']['incorrect'],
                diff8  = copes['difficulty8']['incorrect'],
                diff12 = copes['difficulty12']['incorrect'] ),
                    # correct  = copes['difficulty'+str(difficulty)]['correct'],
                    # incorrect  = copes['difficulty'+str(difficulty)]['incorrect'],
                    # difference = copes['difficulty'+str(difficulty)]['incorrvscorr']),
            colors = dict(
                diff2  = '#e41a1c',
                diff4  = '#377eb8',
                diff8  = '#984ea3',
                diff12 = '#4daf4a'),
                    # correct = '#2ca25f',
                    # incorrect = '#e34a33',
                    # difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'incorrect trials, electrode %s'%(channel), truncate_xaxis = False, axes=ax)
    # plt.xlim([-0.3, 0.5])

fig = plt.figure(figsize = [8,4])
ax = fig.add_subplot(111)
for channel in chan2use:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                diff2  = copes['difficulty2']['correct'],
                diff4  = copes['difficulty4']['correct'],
                diff8  = copes['difficulty8']['correct'],
                diff12 = copes['difficulty12']['correct'] ),
                    # correct  = copes['difficulty'+str(difficulty)]['correct'],
                    # incorrect  = copes['difficulty'+str(difficulty)]['incorrect'],
                    # difference = copes['difficulty'+str(difficulty)]['incorrvscorr']),
            colors = dict(
                diff2  = '#e41a1c',
                diff4  = '#377eb8',
                diff8  = '#984ea3',
                diff12 = '#4daf4a'),
                    # correct = '#2ca25f',
                    # incorrect = '#e34a33',
                    # difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'correct trials, electrode %s'%(channel), truncate_xaxis = False, axes=ax)
    # plt.xlim([-0.3, 0.5])


fig = plt.figure(figsize = [8,4])
ax = fig.add_subplot(111)
for channel in chan2use:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                diff2  = copes['difficulty2']['incorrvscorr'],
                diff4  = copes['difficulty4']['incorrvscorr'],
                diff8  = copes['difficulty8']['incorrvscorr'],
                diff12 = copes['difficulty12']['incorrvscorr'] ),
                    # correct  = copes['difficulty'+str(difficulty)]['correct'],
                    # incorrect  = copes['difficulty'+str(difficulty)]['incorrect'],
                    # difference = copes['difficulty'+str(difficulty)]['incorrvscorr']),
            colors = dict(
                diff2  = '#e41a1c',
                diff4  = '#377eb8',
                diff8  = '#984ea3',
                diff12 = '#4daf4a'),
                    # correct = '#2ca25f',
                    # incorrect = '#e34a33',
                    # difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'incorrect vs correct trials, electrode %s'%(channel), truncate_xaxis = False, axes=ax)
    # plt.xlim([-0.3, 0.5])




