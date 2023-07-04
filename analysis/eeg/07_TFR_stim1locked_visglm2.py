# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:07:24 2023

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

regs = ['d2corr','d4corr','d8corr','d12corr','d2incorr','d4incorr','d8incorr','d12incorr']
contrasts = ['d2corr','d4corr','d8corr','d12corr','d2incorr','d4incorr','d8incorr','d12incorr',
             'correct', 'incorrect', 'grandmean', 'diff2', 'diff4', 'diff8', 'diff12', 'corrvsincorr']

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
            mne.time_frequency.read_tfrs(  fname = op.join(wd, 'glms', 'stim1locked', 'glm2', 
                                          param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_beta-tfr.h5'))[0].crop(tmin=-2.85, tmax=2.85))
            
    for name in contrasts:
        c[name].append(mne.time_frequency.read_tfrs(fname = op.join(wd, 'glms', 'stim1locked', 'glm2', 
                                      param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_cope-tfr.h5'))[0].crop(tmin=-2.85, tmax=2.85))
        t[name].append(mne.time_frequency.read_tfrs(fname = op.join(wd, 'glms', 'stim1locked', 'glm2', 
                                      param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_tstat-tfr.h5'))[0].crop(tmin=-2.85, tmax=2.85))

#%%
alltimes = mne.grand_average(b['d2corr']).times
allfreqs = mne.grand_average(b['d2corr']).freqs
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
for contrast in ['diff2', 'diff4', 'diff8', 'diff12']:#contrasts:
    if contrast in ['grandmean', 'correct', 'incorrect']:
        # baseline = (-2.7, -2.4)
        baseline = None
    else:
        baseline = None
    dat2plot = mne.grand_average(c[contrast])
    dat2plot.plot_joint(baseline = baseline, timefreqs = timefreqs_alpha, topomap_args = topoargs, title = 'cope: '+contrast)

#%%

#get alpha only in the posterior channels and plot timecourse of alpha for each difficulty level
d2 = np.empty(shape = [len(subs), len(alltimes)])
d4 = np.empty(shape = [len(subs), len(alltimes)])
d8 = np.empty(shape = [len(subs), len(alltimes)])
d12 = np.empty(shape = [len(subs), len(alltimes)])
diffs = [d2, d4, d8, d12]
difficulties = [2,4,8,12]

tmp = deepcopy(c)

for idiff in range(len(diffs)):
    tmpdat = c['diff'+str(difficulties[idiff])] #list of averageTFRs per subject
    for sub in range(len(subs)):
        tmpsub = tmpdat[sub].copy().crop(fmin = 8, fmax = 12).pick(picks = posteriorchannels)
        diffs[idiff][sub] = tmpsub.data.mean(axis=1).mean(axis=0) #average across frequencies then channels

#%%
#plot this ?
cols = ['#386cb0', '#beaed4', '#fdc086', '#7fc97f']

fig= plt.figure()
ax = fig.add_subplot(111)
for idiff in range(len(diffs)):
    tmpdat = diffs[idiff]
    ax.plot(alltimes, tmpdat.mean(0), label = 'diff'+str(difficulties[idiff]), color = cols[idiff])
    ax.fill_between(x = alltimes,
                    y1 = np.subtract(tmpdat.mean(0), sp.stats.sem(tmpdat, axis = 0, ddof = 0)),
                    y2 = np.add(tmpdat.mean(0), sp.stats.sem(tmpdat, axis = 0, ddof = 0)),
                    color = cols[idiff], alpha = 0.3)
ax.axvspan(xmin = -2.7, xmax = -2.4, color = '#bdbdbd', alpha = 0.2)
fig.legend()


