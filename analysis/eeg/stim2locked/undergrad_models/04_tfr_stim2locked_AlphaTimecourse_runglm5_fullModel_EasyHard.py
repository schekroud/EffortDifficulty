# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:18:51 2024

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
import glmtools as glm

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,    38, 39])
#drop 36 & 37 as unusable and withdrew from task
glms2run = 1 #1 with no baseline, one where tfr input data is baselined
smooth = False #if smoothing single trial alpha timecourse
transform = False #if converting power to decibels (10*log10 power)
glmdir = op.join(wd, 'glms', 'stim2locked', 'alpha_timecourses', 'undergrad_models', 'glm5')
if not op.exists(glmdir):
    os.mkdir(glmdir)

for i in subs:
    for iglm in [1]: #controls whether glm is run on un-baselined or baselined data
        print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
        sub   = dict(loc = 'workstation', id = i)
        param = getSubjectInfo(sub)
        tfr = mne.time_frequency.read_tfrs(param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
        
        bline_vals = np.load(op.join(wd, 'eeg', 's%02d'%i, 'EffortDifficulty_s%02d_stim1locked_baselineValues_AlphaOnly.npy'%i)) #load baseline vals
        keep_trls = np.array([x for x in np.arange(1, 321) if x in tfr.metadata.trlid.to_numpy()])
        keeptrl_inds = np.subtract(keep_trls, 1) #get index
        
        blines = bline_vals[keeptrl_inds,:,:] #keep just baseline vals for the cleaned trials we want to analyse
        #get zscored trial numbr to include as regressor for time on task (in case it affects alpha estimated)
        tmpdf = pd.DataFrame(np.array([np.add(np.arange(320),1).astype(int), 
                              sp.stats.zscore(np.add(np.arange(320),1))]).T, columns = ['trlid', 'trlid_z'])
        
        tmpdf = tmpdf.query('trlid in @tfr.metadata.trlid').reset_index(drop=True)
        tmpdf2 = tfr.metadata.copy().reset_index(drop=True)
        tmpdf2 = tmpdf2.assign(trlidz = tmpdf.trlid_z)
        tfr.metadata = tmpdf2
        
        if iglm == 0:
            addtopath = ''
            baseline_input = False
        elif iglm == 1:
            addtopath = '_baselined'
            baseline_input = True
           
        posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8', 'Oz', 'POz']
        
        #set baseline params to test stuff
        if baseline_input:
            tfrdata = tfr.data.copy()
            for itime in range(tfr.times.size): #loop over timepoints
                tfrdata[:,:,:,itime] = np.subtract(tfrdata[:,:,:,itime], blines)
            tfr.data = tfrdata #re-assign the baselined data back into the tfr object
        elif not baseline_input:
            bline = None
        
        #comes with metadata attached            
        tfr = tfr['fbtrig != 62'] #drop timeout trials
        tfr = tfr['diffseqpos > 3'] #remove the first three trials of each new difficulty sequence
        #so we only look at trials where they should have realised the difficulty level
        tfr = tfr['prevtrlfb in ["correct", "incorrect"]'] #drop trials where previous trial was a timeout, as this is a big contributor to some effects
        
        if i == 16:
            tfr = tfr['blocknumber != 4']
        if i in [21, 25]:
            #these participants had problems in block 1
            tfr = tfr['blocknumber > 1']
        
        posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8', 'Oz', 'POz']          
        tfrdat = tfr.copy().pick_channels(posterior_channels).data.copy()
        tfrdat = np.mean(tfrdat, axis = 2) #average across the frequency band, results in trials x channels x time
        tfrdat = np.mean(tfrdat, axis = 1) #average across channels now, returns trials x time
        if smooth:
            tfrdat = gauss_smooth(tfrdat, sigma = 2)
        if transform:
            tfrdat = np.multiply(10, np.log10(tfrdat))
        
        trialnum    = tfr.metadata.trlidz.to_numpy()
        correctness = tfr.metadata.PerceptDecCorrect.to_numpy()
        correctness = np.where(correctness == 0, -1, correctness)
        prevtrlfb   = tfr.metadata.prevtrlfb.to_numpy()
        prevtrlfb   = np.where(prevtrlfb == "correct", 1, -1)
        difficulty  = tfr.metadata.difficultyOri.to_numpy()
        
        #put the whole thing into one designmatrix
        d2    = difficulty == 2
        d4    = difficulty == 4
        d8    = difficulty == 8
        d12   = difficulty == 12
        corr  = correctness == 1
        incorr= correctness == -1
        prevcorr   = prevtrlfb == 1
        previncorr = prevtrlfb == -1
        
        hard = d2.copy(); hard |= d4   # combine trials of difficulty 2 or 4 degrees
        easy = d8.copy(); easy |= d12  # combine trials of difficulty 8 or 12 degrees
        
        def nan_zscore(array):
            array = array.astype(float)
            array[array==0] = np.nan
            array = sp.stats.zscore(array, nan_policy='omit')
            array[np.isnan(array)] = 0
            return array
        
        hardcorrectness = nan_zscore(np.multiply(hard, correctness).astype(float))
        easycorrectness = nan_zscore(np.multiply(easy, correctness).astype(float))
        
        hardprevcorrectness = nan_zscore(np.multiply(hard, prevtrlfb).astype(float))
        easyprevcorrectness = nan_zscore(np.multiply(easy, prevtrlfb).astype(float))
                
        DC = glm.design.DesignConfig()
        DC.add_regressor(name = 'hard', rtype = 'Categorical', datainfo = 'hard', codes = 1)
        DC.add_regressor(name = 'easy', rtype = 'Categorical', datainfo = 'easy', codes = 1)
        DC.add_regressor(name = 'hardcorrectness',      rtype = 'Parametric', datainfo = 'hardcorrectness',     preproc = None)
        DC.add_regressor(name = 'easycorrectness',      rtype = 'Parametric', datainfo = 'easycorrectness',     preproc = None)
        DC.add_regressor(name = 'hardprevcorrectness',  rtype = 'Parametric', datainfo = 'hardprevcorrectness',  preproc = None)
        DC.add_regressor(name = 'easyprevcorrectness',  rtype = 'Parametric', datainfo = 'easyprevcorrectness',  preproc = None)
        DC.add_regressor(name = 'trlnum', rtype = 'Parametric', datainfo = 'trialnum', preproc = None)
        DC.add_simple_contrasts()
        DC.add_contrast(values = [1, 0, 1, 0, 0, 0, 0], name = 'hardcorr')
        DC.add_contrast(values = [1, 0,-1, 0, 0, 0, 0], name = 'hardincorr')
        DC.add_contrast(values = [0, 1, 0, 1, 0, 0, 0], name = 'easycorr')
        DC.add_contrast(values = [0, 1, 0,-1, 0, 0, 0], name = 'easyincorr')
        DC.add_contrast(values = [1, 0, 0, 0, 1, 0, 0], name = 'hardprevtrlcorr')
        DC.add_contrast(values = [1, 0, 0, 0,-1, 0, 0], name = 'hardprevtrlincorr')
        DC.add_contrast(values = [0, 1, 0, 0, 0, 1, 0], name = 'easyprevtrlcorr')
        DC.add_contrast(values = [0, 1, 0, 0, 0,-1, 0], name = 'easyprevtrlincorr')        
        

        glmdata = glm.data.TrialGLMData(data = tfrdat, time_dimw = 1, sample_rate = 100,
                                        difficulty = difficulty,
                                        hard = hard,
                                        easy = easy,
                                        hardcorrectness = hardcorrectness,
                                        easycorrectness = easycorrectness,
                                        hardprevcorrectness = hardprevcorrectness,
                                        easyprevcorrectness = easyprevcorrectness,
                                        trialnum = trialnum)
        glmdes = DC.design_from_datainfo(glmdata.info)
        
        if i == 10: #plot example of the design matrix
            fig = plt.figure(figsize = [8,6])
            ax = fig.add_subplot(111)
            ax.imshow(glmdes.design_matrix, aspect= 'auto', vmin = -2, vmax = 2, cmap = 'RdBu_r', interpolation = None)
            ax.set_xticks(range(glmdes.design_matrix.shape[1]), labels = glmdes.regressor_names, rotation = 45)
            ax.set_ylabel('trial number')
            fig.tight_layout()
            fig.savefig(op.join(glmdir, 'example_designmatrix.pdf'), format = 'pdf', dpi = 300)
        
        # glmdes.plot_summary(summary_lines=False)
        # glmdes.plot_efficiency()
        
        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model 
            
        betas = model.betas.copy()
        copes = model.copes.copy()
        tstats = model.tstats.copy()
        
        np.save(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_betas.npy'), arr = betas)
        np.save(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_copes.npy'), arr = copes)
        np.save(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_tstats.npy'), arr = tstats)
        
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