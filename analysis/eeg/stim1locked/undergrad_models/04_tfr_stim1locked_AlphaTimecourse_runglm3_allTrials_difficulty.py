# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:02:58 2024

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
from funcs import getSubjectInfo, gauss_smooth

wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)
import glmtools as glm

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,    38, 39])
#drop 36 & 37 as unusable and withdrew from task
glms2run = 1 #1 with no baseline, one where tfr input data is baselined
smooth = True #if smoothing single trial alpha timecourse
transform = False #if converting power to decibels (10*log10 power)
glmdir = op.join(wd, 'glms', 'stim1locked', 'alpha_timecourses', 'undergrad_models', 'glm3')
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
        tfr = tfr['diffseqpos > 3'] #remove the first three trials of each new difficulty sequence
        #so we only look at trials where they should have realised the difficulty level
        tfr = tfr['prevtrlfb in ["correct", "incorrect"]'] #drop trials where previous trial was a timeout, as this is a big contributor to some effects
        
        if i == 16:
            tfr = tfr['blocknumber != 4']
        if i in [21, 25]:
            #these participants had problems in block 1
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
        
        trialnum  = tfr.metadata.trlidz.to_numpy() #regressor for epoch number (proxy for time on task)
        acc       = tfr.metadata.PerceptDecCorrect.to_numpy()
        acc       = np.where(acc == 1, 1, -1)
        acc       = sp.stats.zscore(acc) #zscore this across trials
        prevtrlfb = tfr.metadata.prevtrlfb.to_numpy()
        prevtrlfb = np.where(prevtrlfb == "correct", 1, -1)
        difficulty = tfr.metadata.difficultyOri.to_numpy()

        DC = glm.design.DesignConfig()
        DC.add_regressor(name = 'diff2',  rtype = 'Categorical', datainfo = 'difficulty', codes = 2)
        DC.add_regressor(name = 'diff4',  rtype = 'Categorical', datainfo = 'difficulty', codes = 4)
        DC.add_regressor(name = 'diff8',  rtype = 'Categorical', datainfo = 'difficulty', codes = 8)
        DC.add_regressor(name = 'diff12', rtype = 'Categorical', datainfo = 'difficulty', codes = 12)
        DC.add_regressor(name = 'trialnumber', rtype = 'Parametric', datainfo = 'trialnum', preproc = None)
        DC.add_simple_contrasts() #add basic diagonal matrix for copes
        #create glmdata object
        glmdata = glm.data.TrialGLMData(data = tfrdat, time_dim = 1, sample_rate = 100,
                                        #add in metadata that's used to construct the design matrix
                                        difficulty = difficulty,
                                        trialnum = trialnum
                                        )
    
        glmdes = DC.design_from_datainfo(glmdata.info)
        
        if i == 10: #plot example of the design matrix
            fig = plt.figure(figsize = [5,4])
            ax = fig.add_subplot(111)
            ax.imshow(glmdes.design_matrix, aspect= 'auto', vmin = -2, vmax = 2, cmap = 'RdBu_r', interpolation = None)
            ax.set_xticks(range(glmdes.design_matrix.shape[1]), labels = glmdes.regressor_names)
            ax.set_ylabel('trial number')
            fig.savefig(op.join(glmdir, 'example_designmatrix.pdf'), format = 'pdf', dpi = 300)
        
        # glmdes.plot_summary(summary_lines=False)
        # glmdes.plot_efficiency()
        
        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model 
            
        betas = model.betas.copy()
        copes = model.copes.copy()
        tstats = model.tstats.copy()
        
        np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_betas.npy'), arr = betas)
        np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_copes.npy'), arr = copes)
        np.save(file = op.join(glmdir, param['subid'] + '_stim1lockedTFR_tstats.npy'), arr = tstats)
        
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info
    
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(times, copes.T, label = model.contrast_names, lw = 1)
        # ax.axvline(x = 0, ls = 'dashed', color = '#000000', lw = 1)
        # ax.axhline(y = 0, ls = 'dashed', color = '#000000', lw = 1)
        # if baseline_input:
        #     ax.axvspan(xmin = bline[0], xmax = bline[1], color = '#bdbdbd', lw = 0, edgecolor = None, alpha = 0.3)
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