#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:27:58 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import os.path as op
from matplotlib import pyplot as plt
import statsmodels as sm
import statsmodels.api as sma
import mne
import os
import sys
import glmtools as glm
%matplotlib
mne.viz.set_browser_backend('qt')
np.set_printoptions(suppress=True)

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo
from eyefuncs_mne import find_blinks, transform_pupil

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

eyedir  = op.join(wd, 'data', 'eyes', 'asc')
datadir = op.join(wd, 'data', 'datafiles', 'combined') #path to single subject datafiles
figdir  = op.join(wd, 'figures', 'eyes') 
trlcheckdir = op.join(wd, 'data', 'eyes', 'trialchecks', 'stim1locked')

drop_gaze = True

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,         31, 32, 33, 34, 35,             39])
# subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now

# subs = np.array([10, 12, 13, 14, 15, 16, 17, 20,  24,     26, 27, 29, 30, 31, 32, 33, 34, 35,             39]) #19 subjects

thetas = np.array([2, 4, 8, 12])
baselined    = False
check_exists = False
save_data    = False
chan2use = 'pupil'

allBetas = np.zeros(shape = [subs.size, thetas.size, 4, 7001])
allCopes = np.zeros(shape = [subs.size, thetas.size, 6, 7001])
icount = -1
for i in subs:
    icount += 1
    print(f"\n- - - - working on participant {i} - - - -")
    sub   = dict(loc = 'laptop', id = i)
    param = getSubjectInfo(sub)
    stim1locked = mne.epochs.read_epochs(param['s1locked_eyes'], preload=True) #read in data
    
    #get zscored trial numbr to include as regressor for time on task (in case it affects alpha estimated)
    tmpdf = pd.DataFrame(np.array([np.add(np.arange(320),1).astype(int), 
                          sp.stats.zscore(np.add(np.arange(320),1))]).T, columns = ['trlid', 'trlid_z'])
    
    tmpdf = tmpdf.query('trlid in @stim1locked.metadata.trlid').reset_index(drop=True)
    tmpdf2 = stim1locked.metadata.copy().reset_index(drop=True)
    tmpdf2 = tmpdf2.assign(trlidz = tmpdf.trlid_z)
    stim1locked.metadata = tmpdf2
    

    #get info on %missing data per trial
    trlnans_perc = np.load(file = op.join(trlcheckdir, f"{param['subid']}_stim1locked_eyes_nancheck_perTrial.npy"))
    to_remove = np.where(trlnans_perc>50)[0] #find trials where blanket over 50% of data is missing to remove
    to_keep = trlnans_perc<30
    
    #discard some trials from this vector based on ones already thrown out
    # to_keep = to_keep[stim1locked.metadata.trlid-1]
    stim1locked.metadata = stim1locked.metadata.assign(tokeep = to_keep)
    stim1locked = stim1locked['tokeep == True']
    
    
    times = stim1locked.times
    
    #remove some relevant trials?
    if i ==16:
        stim1locked = stim1locked['blocknumber != 4'] #lost the eye in block four of this participant
    if i in [21,25]:
        stim1locked = stim1locked['blocknumber > 1'] #these ppts had problems in block 1 of the task
    
    if i == 26:
        stim1locked = stim1locked['trlid != 192'] #this trial has large amounts of interpolated data, and some nans persist
    stim1locked = stim1locked['fbtrig != 62'] #drop timeout trials
    
    baseline = None
    if baselined:
        baseline = [-1, -0.5]
        # baseline = [-0.5, 0]
    stim1locked = stim1locked.apply_baseline(baseline)
    nonchans = [x for x in stim1locked.ch_names if x != chan2use]
    stim1locked = stim1locked.drop_channels(nonchans) #drop everything that we dont need
    
    #take only trials where they should have realised difficulty has changed (i.e. trials 4 onwards in difficulty sequence)
    stim1locked = stim1locked['diffseqpos >= 4']
    stim1locked = stim1locked['prevtrlfb in ["correct","incorrect"]'] #discard trials where they timed out on the previous trial
    pupil_data = np.squeeze(stim1locked._data)
    
    bdata      = stim1locked.metadata
    difficulty = bdata.difficultyOri.to_numpy()
    trialnum   = bdata.trlidz.to_numpy()
    acc        = bdata.PerceptDecCorrect.to_numpy()
    prevtrlfb  = bdata.prevtrlfb.to_numpy()
    
    betas = np.zeros(shape = [np.unique(difficulty).size, 4, times.size])
    copes = np.zeros(shape = [np.unique(difficulty).size, 6, times.size]) #added two contrasts to get corr/incorr trials
    
    diffcount = -1
    for idiff in np.sort(np.unique(difficulty)): #loop over each difficulty to run glm
        diffcount += 1 
        diffids         = np.where(difficulty == idiff)[0]
        tmpdat          = pupil_data[diffids]
        idiff_acc       = acc[diffids]
        idiff_trlnum    = trialnum[diffids]
        idiff_prevtrlfb = prevtrlfb[diffids]
        
        #set up regressors more
        idiff_acc       = np.where(idiff_acc == 1, 1, -1)
        idiff_prevtrlfb = np.where(idiff_prevtrlfb == "correct", 1, -1)
        
        DC = glm.design.DesignConfig()
        DC.add_regressor(name = 'intercept', rtype = 'Constant')
        DC.add_regressor(name = 'accuracy',  rtype = 'Parametric', datainfo = 'idiff_acc',       preproc = None)
        DC.add_regressor(name = 'trialnum',  rtype = 'Parametric', datainfo = 'idiff_trlnum',    preproc = None)
        DC.add_regressor(name = 'prevtrlfb', rtype = 'Parametric', datainfo = 'idiff_prevtrlfb', preproc = None)
        DC.add_simple_contrasts()
        DC.add_contrast(values = [1, 1, 0, 0], name = 'correct')
        DC.add_contrast(values = [1,-1, 0, 0], name = 'incorrect')
        
        glmdata = glm.data.TrialGLMData(data = tmpdat, time_dim = 1, sample_rate = 1000, 
                                        idiff_acc = idiff_acc,
                                        idiff_trlnum = idiff_trlnum,
                                        idiff_prevtrlfb = idiff_prevtrlfb)
        glmdes = DC.design_from_datainfo(glmdata.info)
        
        model = glm.fit.OLSModel(glmdes, glmdata) #fit model
        
        ibetas = model.betas.copy()
        icopes = model.copes.copy()
        
        betas[diffcount] = ibetas
        copes[diffcount] = icopes
    allBetas[icount] = betas
    allCopes[icount] = copes
regnames = glmdes.regressor_names
copenames = glmdes.contrast_names
#%%

means = allBetas.mean(axis=0)
sems  = sp.stats.sem(allBetas, axis = 0, ddof = 0)
#these now have shape: [difficulty, regressor, time]
fig = plt.figure(figsize = [15,15])
for idiff in range(means.shape[0]): #loop over difficulties
    tmpmean, tmpsem = means[idiff], sems[idiff]
    ax = fig.add_subplot(2, 2, idiff+1)
    for ireg in range(tmpmean.shape[0]): #loop over regressors
        if ireg >0:    
            ax.plot(times, tmpmean[ireg], label = regnames[ireg], lw = 2)
            ax.fill_between(times, np.add(tmpmean[ireg], tmpsem[ireg]), np.subtract(tmpmean[ireg], tmpsem[ireg]),
                            alpha = 0.3, edgecolor = None, lw = 0)
            ax.set_title(f"difficulty {thetas[idiff]}")
    
    ax.legend(loc = 'right', bbox_to_anchor = [1.15, 0.75])
    ax.axvline(0, color = '#000000', lw = 1, ls='dashed')
    ax.axhline(0, color = '#000000', lw = 1, ls = 'dashed')
    if baselined:
        ax.axvspan(xmin = baseline[0], xmax = baseline[1], alpha = 0.3, color = '#bdbdbd', edgecolor = None, lw = 0)












#%%
cols = ['#000000','#377eb8','#4daf4a','#984ea3','#ff7f00']
intercept  = np.zeros(shape = (subs.size, times.size))
accuracy   = np.zeros_like(intercept)
difficulty = np.zeros_like(intercept)
prevtrlfb  = np.zeros_like(intercept)
trlnum     = np.zeros_like(intercept)
# ISI        = np.zeros_like(intercept)

for isub in range(subs.size):
    tmpdf = allBetas.query('subid == @subs[@isub]')
    intercept[isub]  = tmpdf.const.to_numpy()
    accuracy[isub]   = tmpdf.accuracy.to_numpy()
    difficulty[isub] = tmpdf.difficulty.to_numpy()
    prevtrlfb[isub]  = tmpdf.prevtrlfb.to_numpy()
    trlnum[isub]     = tmpdf.trlnum.to_numpy()
    # ISI[isub]        = tmpdf.ISI.to_numpy()

intercept_mean, intercept_sem   = np.mean(intercept, axis=0),  sp.stats.sem(intercept, axis = 0, ddof = 0)
accuracy_mean, accuracy_sem     = np.mean(accuracy, axis=0),   sp.stats.sem(accuracy, axis = 0, ddof = 0)
difficulty_mean, difficulty_sem = np.mean(difficulty, axis=0), sp.stats.sem(difficulty, axis = 0, ddof = 0)
prevtrlfb_mean, prevtrlfb_sem   = np.mean(prevtrlfb, axis=0),  sp.stats.sem(prevtrlfb, axis = 0, ddof = 0)
trlnum_mean, trlnum_sem         = np.mean(trlnum, axis=0),     sp.stats.sem(trlnum, axis=0, ddof = 0)
# isi_mean, isi_sem = np.mean(ISI, axis=0), sp.stats.sem(ISI, axis = 0, ddof = 0)


fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
# ax.plot(times, intercept_mean, label = 'intercept', color = cols[0])
# ax.fill_between(times, np.add(intercept_mean, intercept_sem), np.subtract(intercept_mean, intercept_sem),
#                 color = cols[0], alpha = 0.3, lw=0)
ax.plot(times, accuracy_mean, label = 'accuracy', color = cols[1])
ax.fill_between(times, np.add(accuracy_mean, accuracy_sem), np.subtract(accuracy_mean, accuracy_sem),
                color = cols[1], alpha = 0.3, lw=0)
ax.plot(times, difficulty_mean, label = 'difficulty', color = cols[2])
ax.fill_between(times, np.add(difficulty_mean, difficulty_sem), np.subtract(difficulty_mean, difficulty_sem),
                color = cols[2], alpha = 0.3, lw=0)
ax.plot(times, prevtrlfb_mean, label = 'prevtrlfb', color = cols[3])
ax.fill_between(times, np.add(prevtrlfb_mean, prevtrlfb_sem), np.subtract(prevtrlfb_mean, prevtrlfb_sem),
                color = cols[3], alpha = 0.3, lw=0)
ax.plot(times, trlnum_mean, label = 'trlnum', color = cols[4])
ax.fill_between(times, np.add(trlnum_mean, trlnum_sem), np.subtract(trlnum_mean, trlnum_sem),
                color = cols[4], alpha = 0.3, lw=0)
ax.legend(loc = 'right', bbox_to_anchor = [1.15, 0.75])
ax.axvline(0, color = '#000000', lw = 1, ls='dashed')
ax.axhline(0, color = '#000000', lw = 1, ls = 'dashed')
if baselined:
    ax.axvspan(xmin = baseline[0], xmax = baseline[1], alpha = 0.3, color = '#bdbdbd', edgecolor = None, lw = 0)
    
    