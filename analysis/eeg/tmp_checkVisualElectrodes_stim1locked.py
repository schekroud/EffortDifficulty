# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:04:43 2023

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
import glmtools as glm

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')

from funcs import getSubjectInfo

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)

subs = np.array([10, 11, 12, 13, 14, 15, 16])

def gauss_smooth(array, sigma = 2):
    return sp.ndimage.gaussian_filter1d(array, sigma=sigma, axis = 1) #smooths across time, given 2d array of trials x time


#%%
gmean_data = []
for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    
    tfr = mne.time_frequency.read_tfrs(param['stim1locked'].replace('stim1locked', 'stim1locked_cleaned').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
    # tfr = mne.time_frequency.read_tfrs(param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
    
    baseline=None
    #just want to get the average induced response to the stimulus for channel selection, independent of any task or behavioural parameters
    gmean_data.append(tfr.apply_baseline(baseline = baseline).average())

posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8']

#%%
gmean = deepcopy(gmean_data)
timefreqs_alpha = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4),
                   (1., 10):(.4, 4),
                   (1.2, 10):(.4, 4)}
timefreqs_theta = {(.4, 6):(.4, 4),
                   (.6, 6):(.4, 4),
                   (.8, 6):(.4, 4),
                   (1., 6):(.4, 4),
                   (1.2, 6):(.4, 4)}
timefreqs_beta  = {(.4, 20):(.4, 10),
                   (.6, 20):(.4, 10),
                   (.8, 20):(.4, 10),
                   (1., 20):(.4, 10),
                   (1.2, 20):(.4, 10)}

topoargs = dict(outlines= 'head', contours = 0)


# loop over subjects and baseline their average data % change)
for sub in gmean:
    sub = sub.apply_baseline(baseline = (-2.7, -2.4), mode = 'mean')

gave = mne.grand_average(gmean)    

gave.plot_joint(topomap_args = topoargs, timefreqs = timefreqs_alpha)
posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8']
gave.plot(picks = posterior_channels, combine='mean')
#%%

#get log data?
tmp = deepcopy(gmean_data)
for sub in tmp:
    sub = sub.pick_channels(ch_names=posterior_channels)

tmp_data = np.empty(shape = (len(subs), 600))

#get subject data and average across channels within alpha only
for i in range(len(subs)):
    tmp_data[i] = tmp[i].copy().crop(fmin=8, fmax=12).data.mean(axis=0).mean(axis=0)

times = gmean_data[0].times.copy()
log_tmpdata = np.multiply(10, np.log10(tmp_data.copy()))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, log_tmpdata.mean(axis=0), label = '10log10 power', color = '#3182bd', lw = 2)
ax.fill_between(x = times,
                y1 = np.subtract(log_tmpdata.mean(axis=0), sp.stats.sem(log_tmpdata, axis=0, ddof=0)),
                y2 = np.add(log_tmpdata.mean(axis=0), sp.stats.sem(log_tmpdata, axis=0, ddof=0)),
                color = '#3182bd', alpha = 0.2)
ax.axvspan(xmin = -2.7, xmax = -2.4, alpha = 0.1, color = '#000000')
ax.set_title('no baseline')
fig.legend()

#%%

#get tfr for posterior channels, all frequencies. plot raw, baselined, and 10log10 power with contourf


tmp2 = deepcopy(gmean_data)
for sub in tmp2:
    sub = sub.pick_channels(ch_names = posterior_channels)

#get single subject data, average across these channels to get a 40 (freqs) x 600 (time) array to average across subjects
tmp2data = np.empty(shape= (subs.size, 40, 600))
rawdat = tmp2data.copy()
blined = tmp2data.copy()
log_tmp2 = tmp2data.copy()
logbline_tmp2 = tmp2data.copy()


for i in range(len(tmp2)):
    singlesub = tmp2[i].data.copy().mean(axis=0) #average across channels
    singlesub_bline = tmp2[i].copy().apply_baseline((-2.7, -2.4)).data.mean(axis=0)
    rawdat[i] = singlesub
    blined[i] = singlesub_bline
    
log_tmp2 = np.multiply(10, np.log10(rawdat.copy()))
logbline_tmp2 = np.multiply(10, np.log10(rawdat.copy()))
baseperiod = np.logical_and(np.less_equal(times, -2.4), np.greater_equal(times, -2.7))
baseline_val = logbline_tmp2[:,:,baseperiod].mean(axis=2)

for sub in range(rawdat.shape[0]):
    for freq in range(rawdat.shape[1]):
        for t in range(rawdat.shape[2]):
            logbline_tmp2[sub, freq, t] = logbline_tmp2[sub, freq, t] - baseline_val[sub, freq]

plt.figure(); plt.imshow(rawdat.mean(0), aspect='auto', origin = 'lower')
plt.figure(); plt.imshow(logbline_tmp2.mean(0), aspect='auto', origin = 'lower')

#%%
fig = plt.figure()
ax = fig.add_subplot(221)
axplot = ax.imshow(rawdat.mean(axis=0), cmap = 'RdBu_r', aspect = 'auto', interpolation = 'none', origin = 'lower',
          extent = (times.min(), times.max(), 1, 40))
ax.vlines(ymin = 1, ymax = 40, x = 0, ls = 'dashed', color = 'k', lw = 1)
ax.set_title('raw power')
plt.colorbar(axplot, ax = ax)

ax2 = fig.add_subplot(222)
ax2plot = ax2.imshow(blined.mean(axis=0), cmap = 'RdBu_r', aspect = 'auto', interpolation = 'none', origin = 'lower',
          extent = (times.min(), times.max(), 1, 40))
ax2.vlines(ymin = 1, ymax = 40, x = 0, ls = 'dashed', color = 'k', lw = 1)
ax2.axvspan(xmin = -2.7, xmax = -2.4, color = '#bdbdbd', alpha = 0.3, edgecolor = 'none')
ax2.set_title('baselined power')
plt.colorbar(ax2plot, ax = ax2)


ax3 = fig.add_subplot(223)
ax3plot = ax3.imshow(log_tmp2.mean(axis=0), cmap = 'RdBu_r', aspect = 'auto', interpolation = 'none', origin = 'lower',
          extent = (times.min(), times.max(), 1, 40))
ax3.vlines(ymin = 1, ymax = 40, x = 0, ls = 'dashed', color = 'k', lw = 1)
ax3.set_title('10log10 of raw power')
plt.colorbar(ax3plot, ax = ax3)

ax4 = fig.add_subplot(224)
ax4plot = ax4.imshow(logbline_tmp2.mean(axis=0), cmap = 'RdBu_r', aspect = 'auto', interpolation = 'none', origin = 'lower',
          extent = (times.min(), times.max(), 1, 40))
ax4.vlines(ymin = 1, ymax = 40, x = 0, ls = 'dashed', color = 'k', lw = 1)
ax4.axvspan(xmin = -2.7, xmax = -2.4, color = '#bdbdbd', alpha = 0.3, edgecolor = 'none')
ax4.set_title('baselined 10log10 of raw power')
plt.colorbar(ax4plot, ax = ax4)
