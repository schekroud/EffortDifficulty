#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 18:39:24 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import os.path as op
import statsmodels as sm
import statsmodels.api as sma
from matplotlib import pyplot as plt
import glmtools as glm
import mne
import os
import sys
%matplotlib
mne.viz.set_browser_backend('qt')
np.set_printoptions(suppress=True)

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gauss_smooth
from eyefuncs_mne import find_blinks, transform_pupil

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

eyedir  = op.join(wd, 'data', 'eyes', 'asc')
datadir = op.join(wd, 'data', 'datafiles', 'combined') #path to single subject datafiles
figdir  = op.join(wd, 'figures', 'eyes') 
trlcheckdir = op.join(wd, 'data', 'eyes', 'trialchecks', 'stim1locked')

glmnum = 'glm4'
glmdir = op.join(wd, 'glms', 'eyes', 'stim1locked', 'undergrad_models', glmnum)
if not op.exists(glmdir):
    os.mkdir(glmdir)
subs   = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs   = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])

# subs   = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,      22, 23, 24,     26, 27, 28,        31, 32, 33, 34, 35,             39])
#21 sleepy, 25 issues with collection, 29, 30 recorded in two sessions (technical issues)

smoothing        = True   #smooth the individual trial epoch
baselined        = True 
baseline_perc    = False
# chan2use = 'pupil_transformed'
chan2use = 'pupil'
for iglm in [0, 1]:
    addtopath = ''
    if iglm == 1:
        addtopath = '_nodiff2'
    icount = -1
    for i in subs:
        icount += 1
        print(f"\n- - - - working on participant {i} - - - -")
        sub   = dict(loc = 'laptop', id = i)
        param = getSubjectInfo(sub)
        epochs = mne.epochs.read_epochs(param['s1locked_eyes'], preload=True) #read in data
        times = epochs.times
        
        #keep only channels that we are actually interested in
        chans2drop = [x for x in epochs.ch_names if x != chan2use]
        epochs = epochs.drop_channels(chans2drop)
        
        
        #get zscored trial numbr to include as regressor for time on task (in case it affects alpha estimated)
        tmpdf = pd.DataFrame(np.array([np.add(np.arange(320),1).astype(int), 
                              sp.stats.zscore(np.add(np.arange(320),1))]).T, columns = ['trlid', 'trlid_z'])
        
        tmpdf = tmpdf.query('trlid in @epochs.metadata.trlid').reset_index(drop=True)
        tmpdf2 = epochs.metadata.copy().reset_index(drop=True)
        tmpdf2 = tmpdf2.assign(trlidz = tmpdf.trlid_z)
        epochs.metadata = tmpdf2
        
        #get info on %missing data per trial
        trlnans_perc = np.load(file = op.join(trlcheckdir, f"{param['subid']}_stim1locked_eyes_nancheck_perTrial.npy"))
        to_remove = np.where(trlnans_perc>50)[0] #find trials where blanket over 50% of data is missing to remove
        to_keep = trlnans_perc<30
        
        #discard some trials from this vector based on ones already thrown out
        # to_keep = to_keep[stim1locked.metadata.trlid-1]
        epochs.metadata = epochs.metadata.assign(tokeep = to_keep)
        epochs = epochs['tokeep == True'] #keep only trials with at least 70% of the trial not contaminated by blinks
        
        #comes with metadata attached            
        epochs = epochs['fbtrig != 62'] #drop timeout trials
        epochs = epochs['diffseqpos > 3'] #remove the first three trials of each new difficulty sequence
        #so we only look at trials where they should have realised the difficulty level
        epochs = epochs['prevtrlfb in ["correct", "incorrect"]'] #drop trials where previous trial was a timeout, as this is a big contributor to some effects
        
        if i == 16:
            epochs = epochs['blocknumber != 4']
        if i in [21, 25]:
            #these participants had problems in block 1
            epochs = epochs['blocknumber > 1']
        if i == 26:
            epochs = epochs['trlid != 192'] #this trial has large amounts of interpolated data, and some nans persist
        
        #take only trials where they should have realised difficulty has changed (i.e. trials 4 onwards in difficulty sequence)
        # epochs = epochs['difficultyOri != 2']
        
        baseline = None
        if baselined:
            baseline = [-1, -0.5]
            # baseline = [-1, 0]
            # baseline = [-0.5, 0]
            # baseline = [0, 0.5] #baseline during stim1 onset to account for purely visual modulation?
            if not baseline_perc:
                epochs = epochs.apply_baseline(baseline) #baseline the data if there is a baseline to be applied
        
        
        eyedat = np.squeeze(epochs._data) #get data
        usedat = eyedat.copy() #specify for now we are using the data as a whole
        
        if baselined and baseline_perc:
            timeinds = (times >= baseline[0]) & (times <= baseline[1])
            bline    = eyedat[:,timeinds].copy().mean(axis=1).reshape(-1,1) #this is the mean pupil size in the baseline window
            blined = eyedat.copy()
            blined = np.subtract(blined, bline)
            blined = np.divide(blined, bline) * 100
            usedat = blined.copy()
        
        if smoothing:
            usedat = gauss_smooth(usedat, sigma = 50)
        bdata  = epochs.metadata.copy()
        
        #set up design matrix
        trialnum    = epochs.metadata.trlidz.to_numpy()
        correctness = epochs.metadata.PerceptDecCorrect.to_numpy()
        correctness = np.where(correctness == 0, -1, correctness)
        prevtrlfb   = epochs.metadata.prevtrlfb.to_numpy()
        prevtrlfb   = np.where(prevtrlfb == "correct", 1, -1)
        difficulty  = epochs.metadata.difficultyOri.to_numpy()
        
        #put the whole thing into one designmatrix
        d2    = difficulty == 2
        d4    = difficulty == 4
        d8    = difficulty == 8
        d12   = difficulty == 12
        corr  = correctness == 1
        incorr= correctness == -1
        prevcorr   = prevtrlfb == 1
        previncorr = prevtrlfb == -1
        
        def nan_zscore(array):
            array = array.astype(float)
            array[array==0] = np.nan
            array = sp.stats.zscore(array, nan_policy='omit')
            array[np.isnan(array)] = 0
            return array
        
        d2correctness = nan_zscore(np.multiply(d2, correctness).astype(float))
        d4correctness = nan_zscore(np.multiply(d4, correctness).astype(float))
        d8correctness = nan_zscore(np.multiply(d8, correctness).astype(float))
        d12correctness = nan_zscore(np.multiply(d12, correctness).astype(float))
        
        d2prevcorrectness = nan_zscore(np.multiply(d2, prevtrlfb).astype(float))
        d4prevcorrectness = nan_zscore(np.multiply(d4, prevtrlfb).astype(float))
        d8prevcorrectness = nan_zscore(np.multiply(d8, prevtrlfb).astype(float))
        d12prevcorrectness = nan_zscore(np.multiply(d12, prevtrlfb).astype(float))
        
        if iglm == 0:
            DC = glm.design.DesignConfig()
            DC.add_regressor(name = 'diff2', rtype = 'Categorical', datainfo = 'difficulty', codes = 2)
            DC.add_regressor(name = 'diff4', rtype = 'Categorical', datainfo = 'difficulty', codes = 4)
            DC.add_regressor(name = 'diff8', rtype = 'Categorical', datainfo = 'difficulty', codes = 8)
            DC.add_regressor(name = 'diff12', rtype = 'Categorical', datainfo = 'difficulty', codes = 12)
            DC.add_regressor(name = 'd2correctness',  rtype = 'Parametric', datainfo = 'd2correctness',  preproc = None)
            DC.add_regressor(name = 'd4correctness',  rtype = 'Parametric', datainfo = 'd4correctness',  preproc = None)
            DC.add_regressor(name = 'd8correctness',  rtype = 'Parametric', datainfo = 'd8correctness',  preproc = None)
            DC.add_regressor(name = 'd12correctness', rtype = 'Parametric', datainfo = 'd12correctness', preproc = None)
            DC.add_regressor(name = 'd2prevcorrectness',  rtype = 'Parametric', datainfo = 'd2prevcorrectness',  preproc = None)
            DC.add_regressor(name = 'd4prevcorrectness',  rtype = 'Parametric', datainfo = 'd4prevcorrectness',  preproc = None)
            DC.add_regressor(name = 'd8prevcorrectness',  rtype = 'Parametric', datainfo = 'd8prevcorrectness',  preproc = None)
            DC.add_regressor(name = 'd12prevcorrectness', rtype = 'Parametric', datainfo = 'd12prevcorrectness', preproc = None)
            DC.add_regressor(name = 'trlnum', rtype = 'Parametric', datainfo = 'trialnum', preproc = None)
            DC.add_simple_contrasts()
            DC.add_contrast(values = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], name = 'd2corr')
            DC.add_contrast(values = [1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0], name = 'd2incorr')
            DC.add_contrast(values = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], name = 'd4corr')
            DC.add_contrast(values = [0, 1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0], name = 'd4incorr')
            DC.add_contrast(values = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], name = 'd8corr')
            DC.add_contrast(values = [0, 0, 1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0], name = 'd8incorr')
            DC.add_contrast(values = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], name = 'd12corr')
            DC.add_contrast(values = [0, 0, 0, 1, 0, 0, 0,-1, 0, 0, 0, 0, 0], name = 'd12incorr')
            DC.add_contrast(values = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], name = 'd2prevtrlcorr')
            DC.add_contrast(values = [1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0], name = 'd2prevtrlincorr')
            DC.add_contrast(values = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], name = 'd4prevtrlcorr')
            DC.add_contrast(values = [0, 1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0], name = 'd4prevtrlincorr')
            DC.add_contrast(values = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], name = 'd8prevtrlcorr')
            DC.add_contrast(values = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0], name = 'd8prevtrlincorr')
            DC.add_contrast(values = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], name = 'd12prevtrlcorr')
            DC.add_contrast(values = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,-1, 0], name = 'd12prevtrlincorr')
        elif iglm == 1:
            DC = glm.design.DesignConfig()
            DC.add_regressor(name = 'diff4', rtype = 'Categorical', datainfo = 'difficulty', codes = 4)
            DC.add_regressor(name = 'diff8', rtype = 'Categorical', datainfo = 'difficulty', codes = 8)
            DC.add_regressor(name = 'diff12', rtype = 'Categorical', datainfo = 'difficulty', codes = 12)
            DC.add_regressor(name = 'd4correctness',  rtype = 'Parametric', datainfo = 'd4correctness',  preproc = None)
            DC.add_regressor(name = 'd8correctness',  rtype = 'Parametric', datainfo = 'd8correctness',  preproc = None)
            DC.add_regressor(name = 'd12correctness', rtype = 'Parametric', datainfo = 'd12correctness', preproc = None)
            DC.add_regressor(name = 'd4prevcorrectness',  rtype = 'Parametric', datainfo = 'd4prevcorrectness',  preproc = None)
            DC.add_regressor(name = 'd8prevcorrectness',  rtype = 'Parametric', datainfo = 'd8prevcorrectness',  preproc = None)
            DC.add_regressor(name = 'd12prevcorrectness', rtype = 'Parametric', datainfo = 'd12prevcorrectness', preproc = None)
            DC.add_regressor(name = 'trlnum', rtype = 'Parametric', datainfo = 'trialnum', preproc = None)
            DC.add_simple_contrasts()
            DC.add_contrast(values = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0], name = 'd4corr')
            DC.add_contrast(values = [1, 0, 0,-1, 0, 0, 0, 0, 0, 0], name = 'd4incorr')
            DC.add_contrast(values = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], name = 'd8corr')
            DC.add_contrast(values = [0, 1, 0, 0,-1, 0, 0, 0, 0, 0], name = 'd8incorr')
            DC.add_contrast(values = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0], name = 'd12corr')
            DC.add_contrast(values = [0, 0, 1, 0, 0,-1, 0, 0, 0, 0], name = 'd12incorr')
            DC.add_contrast(values = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0], name = 'd4prevtrlcorr')
            DC.add_contrast(values = [1, 0, 0, 0, 0, 0,-1, 0, 0, 0], name = 'd4prevtrlincorr')
            DC.add_contrast(values = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0], name = 'd8prevtrlcorr')
            DC.add_contrast(values = [0, 1, 0, 0, 0, 0, 0,-1, 0, 0], name = 'd8prevtrlincorr')
            DC.add_contrast(values = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0], name = 'd12prevtrlcorr')
            DC.add_contrast(values = [0, 0, 1, 0, 0, 0, 0, 0,-1, 0], name = 'd12prevtrlincorr')
        
    
        glmdata = glm.data.TrialGLMData(data = usedat, time_dimw = 1, sample_rate = 1000,
                                        difficulty = difficulty,
                                        d2correctness = d2correctness,
                                        d4correctness = d4correctness,
                                        d8correctness = d8correctness,
                                        d12correctness = d12correctness,
                                        d2prevcorrectness = d2prevcorrectness,
                                        d4prevcorrectness = d4prevcorrectness,
                                        d8prevcorrectness = d8prevcorrectness,
                                        d12prevcorrectness = d12prevcorrectness,
                                        trialnum = trialnum)
        glmdes = DC.design_from_datainfo(glmdata.info)
    
        if i == 10: #plot design matrix example
            fig = plt.figure(figsize = [5,5])
            ax = fig.add_subplot(111)
            ax.imshow(glmdes.design_matrix, aspect = 'auto', vmin = -2, vmax = 2, cmap = 'RdBu_r')
            ax.set_ylabel('trial number')
            ax.set_xticks(range(glmdes.design_matrix.shape[1]), glmdes.regressor_names)
            fig.savefig(op.join(glmdir, 'example_designmatrix%s.pdf'%addtopath), format = 'pdf', dpi = 300)
            plt.close('all')
        
        # glmdes.plot_summary(summary_lines=False)
        # glmdes.plot_efficiency()
        
        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model 
            
        betas = model.betas.copy()
        copes = model.copes.copy()
        tstats = model.tstats.copy()
        
        np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedEyes_betas%s.npy'%addtopath), arr = betas)
        np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedEyes_copes%s.npy'%addtopath), arr = copes)
        np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedEyes_tstats%s.npy'%addtopath), arr = tstats)
        
        times = epochs.times
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(times, copes.T, label = model.contrast_names, lw = 1)
        # ax.axvline(x = 0, ls = 'dashed', color = '#000000', lw = 1)
        # ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
        # # if baseline_input:
        # #     ax.axvspan(xmin = bline[0], xmax = bline[1], color = '#bdbdbd', lw = 0, edgecolor = None, alpha = 0.3)
        # fig.legend()
        
        
        
        if i == 10: #for first subject, lets also save a couple things for this glm to help with visualising stuff
            #going to save the times
            np.save(file = op.join(glmdir, 'glm_timerange%s.npy'%addtopath), arr= times)
            #save regressor names and contrast names in the order they are in, to help know what is what
            np.save(file = op.join(glmdir, 'regressor_names%s.npy'%addtopath), arr = model.regressor_names)
            np.save(file = op.join(glmdir, 'contrast_names%s.npy'%addtopath),  arr = model.contrast_names)
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdata)
        del(glmdes)
        del(model)
        del(epochs)