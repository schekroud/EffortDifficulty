#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:32:06 2024

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


baselined    = False
check_exists = False
save_data    = False
chan2use = 'pupil'

if baselined:
    fname = op.join(wd, 'glms', 'eyes', 'stim1locked_modelPupil_byTask_perTimepoint_allBetas_baselined.csv')
else:
    fname = op.join(wd, 'glms', 'eyes', 'stim1locked_modelPupil_byTask_perTimepoint_allBetas.csv')
if op.exists(fname) and check_exists:
    allBetas = pd.read_csv(fname)
else:
    allBetas = pd.DataFrame()
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
        
        bdata = stim1locked.metadata
        
        
        # betas = pd.DataFrame(columns=['intercept', 'accuracy', 'difficulty', 'prevtrlfb', 'ISI', 'time'])
        # betas = pd.DataFrame(columns=['intercept', 'accuracy', 'difficulty', 'prevtrlfb', 'time'])
        betas = pd.DataFrame()
        model_diff_separately = True
        for tp in range(times.size): #loop over time points
            pupil = pupil_data[:,tp].copy() #get across trial pupil diameter for this time point
            
            if not model_diff_separately:
                #set up design matrix - modelling pupil diameter as a function of task structures
                add_intercept = True #do you want to add a constant to the model
                trlnum     = bdata.trlidz.to_numpy()
                difficulty = bdata.difficultyOri.to_numpy()
                acc        = bdata.PerceptDecCorrect.to_numpy()
                prevtrlfb  = bdata.prevtrlfb.to_numpy()
                
                # change some regressors for the design matrix
                acc = np.where(acc == 1, 1, -1) # set as contrast [1,-1] regressor for correct - incorrect
                acc = sp.stats.zscore(acc) #zscore this too
                difficulty = sp.stats.zscore(difficulty) #zcore this
                prevtrlfb = np.where(prevtrlfb == "correct", 1, -1)
                prevtrlfb = sp.stats.zscore(prevtrlfb) #zscore this also, so that other regressors are estimated controlling for previous trial performance
                desmat = np.array([acc, difficulty, prevtrlfb, trlnum]).T#
                dm = pd.DataFrame(desmat, columns = ['accuracy', 'difficulty', 'prevtrlfb', 'trlnum'])#, 'ISI'])
                if add_intercept:
                    dm = sma.add_constant(dm)
                # plt.imshow(dm, aspect='auto', vmin = -1, vmax = 1, cmap='RdBu_r', interpolation=None) #can see the design matrix here
                # plt.xticks(range(dm.shape[1]), labels = dm.columns)
                
                #set up linear regression to model pupil diameter as a function of task parameters
                lm = sm.regression.linear_model.OLS(endog = pupil, exog = dm, missing = 'drop').fit() #if any missing pupil points, drop from design matrix and ignore
                
                #get betas
                tpbetas = pd.DataFrame(lm.params).T
                tpbetas = tpbetas.assign(time = times[tp])
                betas = pd.concat([betas, tpbetas])
            elif model_diff_separately:
                difficulty = bdata.difficultyOri.to_numpy()
                diffs = np.unique(difficulty)
                for idiff in diffs:
                    tmpbdata = bdata.query('difficultyOri == @idiff')
                    diffinds = np.where(difficulty == idiff)[0] #get trials for this difficulty
                    tpdat = pupil[diffinds] #get pupil data for just these trials
                    tmpbdata = bdata.iloc[diffinds]
                    
                    add_intercept = True
                    #get regressors
                    acc = tmpbdata.PerceptDecCorrect.to_numpy()
                    prevtrlfb = tmpbdata.prevtrlfb.to_numpy()
                    trlnum = tmpbdata.trlidz.to_numpy()
                    
                    #standardise
                    acc = sp.stats.zscore(np.where(acc == 1, 1, -1))
                    prevtrlfb = sp.stats.zscore(np.where(prevtrlfb == "correct", 1, -1))
                    
                    dm = pd.DataFrame(np.array([acc, prevtrlfb, trlnum]).T, columns = ['accuracy', 'prevtrlfb', 'trlnum'])
                    if add_intercept:
                        dm = sma.add_constant(dm)
                    # plt.imshow(dm, aspect='auto', vmin = -1, vmax = 1, cmap='RdBu_r', interpolation=None) #can see the design matrix here
                    # plt.xticks(range(dm.shape[1]), labels = dm.columns)
                    
                    lm = sm.regression.linear_model.OLS(endog = tpdat, exog = dm, missing = 'drop').fit() #if any missing pupil points, drop from design matrix and ignore
                
                    tpdiffbetas = pd.DataFrame(lm.params).T
                    tpdiffbetas = tpdiffbetas.assign(time = times[tp], diff = idiff)
                    betas = pd.concat([betas, tpdiffbetas])
                    
                
                
        # cols = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
        # fig = plt.figure(figsize = [6,4])
        # ax = fig.add_subplot(111)
        # # ax.plot(times, betas.intercept, label = 'intercept', color = cols[0])
        # ax.plot(times, betas.accuracy, label = 'accuracy', color = cols[1])
        # ax.plot(times, betas.difficulty, label = 'difficulty', color = cols[2])
        # ax.plot(times, betas.prevtrlfb, label = 'prevtrlfb', color = cols[3])
        # ax.plot(times, betas.ISI, label = 'ISI', color = cols[4])
        # ax.legend(loc = 'right', bbox_to_anchor = (1,0.75))
        
        betas = betas.assign(subid = i)
        allBetas = pd.concat([allBetas, betas])
    
    if save_data:
        allBetas.to_csv(fname, index=False)
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
    
    
    
    
    
    
    
    