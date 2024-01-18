# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:48:55 2023

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
import glmtools as glm
%matplotlib

sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gesd, plot_AR

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
glmdir = op.join(wd, 'glms', 'fblocked', 'glm2')
if not op.exists(glmdir):
    os.mkdir(glmdir)

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
for i in subs:
    print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    epoched = mne.read_epochs(fname = param['fblocked'].replace('fblocked', 'fblocked_cleaned'))
    
    #based on photodiode testing, there is a 25ms delay from trigger onset to maximal photodiode onset, so lets adjust times here
    # epoched.shift_time(tshift = -0.025, relative = True)
    
    epoched.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
    epoched.resample(500) #resample to 500Hz
    
    if i==21:
        #this ppt was sleepy in block 1, which massively drags down the average performance across other blocks (where performance was ok)
        #drop this block
        epoched = epoched['blocknumber > 1']
    
    if i == 25:
        #problem with the keyboard in block 1, drags down average performance across other blocks for one of the conditions
        epoched = epoched['blocknumber > 1']
    
    epoched = epoched['fbtrig != 62']
    
    
    trials = np.ones(len(epoched.metadata)).astype(int)
    # trialnum = epoched.metadata.trlidz.to_numpy()
    correctness = epoched.metadata.PerceptDecCorrect.to_numpy()
    correct = epoched.metadata.rewarded.to_numpy()
    incorrect = epoched.metadata.unrewarded.to_numpy()
    correctness = np.where(correctness == 0, -1, correctness)
    difficulty = epoched.metadata.difficultyOri.to_numpy()
    
    for idiff in np.sort(np.unique(difficulty)): #loop over each level of difficulty in the task and run glm
        diffids = np.where(difficulty == idiff)[0]
        tmpdat = epoched.copy()[diffids]# #get data for just trials of this difficulty
        idifftrls = np.ones(len(diffids)) #intercept for this difficulty
        idiff_correctness = correctness[diffids] #get contrast regressor [corr, -incorr] for just this difficulty
        idiff_rewarded = correct[diffids]
        idiff_unrewarded = incorrect[diffids]
        # idiff_trlidz = trialnum[diffids]
    
        #set up design matrix for this difficulty level
        DC = glm.design.DesignConfig()
        DC.add_regressor(name = 'correct', rtype = 'Categorical', datainfo = 'idiff_rewarded', codes = 1)
        DC.add_regressor(name = 'incorrect', rtype = 'Categorical', datainfo = 'idiff_unrewarded', codes = 1)
        DC.add_simple_contrasts()
        DC.add_contrast(values = [-1, 1], name = 'incorrvscorr')
        DC.add_contrast(values = [ 1, 1], name = 'grandmean')
        
        glmdata = glm.data.TrialGLMData(data = tmpdat.get_data(), time_dim = 2, sample_rate = 500,
                                        idiff_rewarded = idiff_rewarded,
                                        idiff_unrewarded  = idiff_unrewarded)
        glmdes = DC.design_from_datainfo(glmdata.info)
        
    
        # glmdes.plot_summary(summary_lines=False)
        # glmdes.plot_efficiency()
        
        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model on this difficulty data
        
        betas = model.betas.copy()
        copes = model.copes.copy()
        tstats = model.tstats.copy()
        
        total_nave = len(diffids)
        corrnave = sum(idiff_rewarded == 1)
        incorrnave = sum(idiff_unrewarded == 1)
        info = tmpdat.info
        tmin = tmpdat.times.min()
        naves = [corrnave, incorrnave, total_nave, total_nave]
        
        
        for ireg in range(len(glmdes.regressor_names)):
            name = glmdes.regressor_names[ireg]
            nave = naves[ireg]
            tl_beta = mne.EvokedArray(info = info, nave = nave, tmin = tmin, data = betas[ireg])
            tl_beta.save(fname = op.join(glmdir, param['subid']+'_fblockedEpoched_' + name +'_' + 'difficulty' + str(idiff) + '_betas-ave.fif'), overwrite=True)
        
        for icope in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[icope]
            nave = naves[icope]
            
            tl_cope = mne.EvokedArray(info = info, nave = nave, tmin = tmin, data = copes[icope])
            tl_tstat = mne.EvokedArray(info = info, nave = nave, tmin = tmin, data = tstats[icope])
            
            tl_cope.save(fname  = op.join(glmdir, param['subid']+'_fblockedEpoched_' + name +'_' + 'difficulty' + str(idiff) + '_copes-ave.fif'), overwrite=True)
            tl_tstat.save(fname = op.join(glmdir, param['subid']+'_fblockedEpoched_' + name +'_' + 'difficulty' + str(idiff) + '_tstats-ave.fif'), overwrite=True)
                         
        del(tl_beta)
        del(tl_tstat)
        del(tl_cope)

        if i == 10: #for first subject, lets also save a couple things for this glm to help with visualising stuff
            #going to save the times
            np.save(file = op.join(glmdir, 'glm_timerange.npy'), arr= epoched.times)
            #save regressor names and contrast names in the order they are in, to help know what is what
            np.save(file = op.join(glmdir, 'regressor_names.npy'), arr = model.regressor_names)
            np.save(file = op.join(glmdir, 'contrast_names.npy'),  arr = model.contrast_names)