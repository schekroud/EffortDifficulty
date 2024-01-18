# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:37:07 2023

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


subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])

glms2run = 1 #1 with no baseline, one where tfr input data is baselined
smooth = False #if smoothing single trial alpha timecourse
transform = False #if converting power to decibels (10*log10 power)
glmdir = op.join(wd, 'glms', 'stim1locked', 'alpha_timecourses', 'glm4')
if not op.exists(glmdir):
    os.mkdir(glmdir)

for i in subs:
    for iglm in [1]: #controls whether glm is run on un-baselined or baselined data
        print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
        sub   = dict(loc = 'workstation', id = i)
        param = getSubjectInfo(sub)
        tfr = mne.time_frequency.read_tfrs(param['stim1locked'].replace('stim1locked', 'stim1locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
        
        #get zscored trial numbr to include as regressor for time on task (in case it affects alpha estimated)
        tmpdf = pd.DataFrame(np.array([np.add(np.arange(320),1).astype(int), 
                              sp.stats.zscore(np.add(np.arange(320),1))]).T, columns = ['trlid', 'trlid_z'])
        
        tmpdf = tmpdf.query('trlid in @tfr.metadata.trlid').reset_index(drop=True)
        tmpdf2 = tfr.metadata.copy().reset_index(drop=True)
        tmpdf2 = tmpdf2.assign(trlidz = tmpdf.trlid_z)
        tfr.metadata = tmpdf2
        
        
        #comes with metadata attached            
        tfr = tfr['fbtrig != 62'] #drop timeout trials
    
        if i==21:
            #this ppt was sleepy in block 1, which massively drags down the average performance across other blocks (where performance was ok)
            #drop this block
            tfr = tfr['blocknumber > 1']
        
        if i == 25:
            #problem with the keyboard in block 1, drags down average performance across other blocks for one of the conditions
            tfr = tfr['blocknumber > 1']
        
        if iglm == 0:
            addtopath = ''
            baseline_input = False
        elif iglm == 1:
            addtopath = '_baselined'
            baseline_input = True
           
        posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8', 'Oz', 'POz']
        
        #set baseline params to test stuff
        if baseline_input:
            bline = (None, None)
            bline = (-2.7, -2.4) #baseline period for stim1locked
        elif not baseline_input:
            bline = None
            
        tfrdat = tfr.copy().apply_baseline(baseline = bline).pick_channels(posterior_channels).data.copy()
        tfrdat = np.mean(tfrdat, axis = 2) #average across the frequency band, results in trials x channels x time
        tfrdat = np.mean(tfrdat, axis = 1) #average across channels now, returns trials x time
        
        if smooth:
            tfrdat = gauss_smooth(tfrdat, sigma = 2)
        
        if transform:
            tfrdat = np.multiply(10, np.log10(tfrdat))
        
        trials = np.ones(len(tfr.metadata)).astype(int)
        trialnum = tfr.metadata.trlidz.to_numpy()
        correctness = tfr.metadata.PerceptDecCorrect.to_numpy()
        correct = tfr.metadata.rewarded.to_numpy()
        incorrect = tfr.metadata.unrewarded.to_numpy()
        correctness = np.where(correctness == 0, -1, correctness)
        difficulty = tfr.metadata.difficultyOri.to_numpy()
        
        for idiff in np.sort(np.unique(difficulty)): #loop over each level of difficulty in the task and run glm
            diffids = np.where(difficulty == idiff)[0]
            tmpdat = tfrdat[diffids] #get data for just trials of this difficulty
            idifftrls = np.ones(len(diffids)) #intercept for this difficulty
            idiff_correctness = correctness[diffids] #get contrast regressor [corr, -incorr] for just this difficulty
            idiff_trlidz = trialnum[diffids]
        
            #set up design matrix for this difficulty level
            DC = glm.design.DesignConfig()
            DC.add_regressor(name = 'intercept', rtype = 'Constant')
            DC.add_regressor(name = 'correctness', rtype = 'Parametric', datainfo = 'idiff_correctness', preproc = None)
            DC.add_regressor(name = 'trialnumber', rtype = 'Parametric', datainfo = 'trialnumber', preproc= None)
            DC.add_simple_contrasts()
            DC.add_contrast(values = [1, 1, 0], name = 'correct')
            DC.add_contrast(values = [1,-1, 0], name = 'incorrect')
            
            glmdata = glm.data.TrialGLMData(data = tmpdat, time_dim = 1, sample_rate = 100,
                                            idiff_correctness = idiff_correctness,
                                            trialnumber = idiff_trlidz)
            glmdes = DC.design_from_datainfo(glmdata.info)
            
        
            # glmdes.plot_summary(summary_lines=False)
            # glmdes.plot_efficiency()
            
            print('\n - - - - -  running glm - - - - - \n')
            model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model on this difficulty data
            
            betas = model.betas.copy()
            copes = model.copes.copy()
            tstats = model.tstats.copy()
            
            np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_betas_difficulty%s.npy'%str(idiff)), arr = betas)
            np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_copes_difficulty%s.npy'%str(idiff)), arr = copes)
            np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_tstats_difficulty%s.npy'%str(idiff)), arr = tstats)
            
            times = tfr.times
            freqs = tfr.freqs
            info = tfr.info
    
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(times, copes.T, label = model.contrast_names, lw = 1)
        # ax.axvline(x = 0, ls = 'dashed', color = '#000000', lw = 1)
        # ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
        # fig.legend()
    
    
    
        if i == 10: #for first subject, lets also save a couple things for this glm to help with visualising stuff
            #going to save the times
            np.save(file = op.join(glmdir, 'glm_timerange.npy'), arr= times)
            #save regressor names and contrast names in the order they are in, to help know what is what
            np.save(file = op.join(glmdir, 'regressor_names.npy'), arr = model.regressor_names)
            np.save(file = op.join(glmdir, 'contrast_names.npy'),  arr = model.contrast_names)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    del(glmdata)
    del(glmdes)
    del(model)
    del(tfr)