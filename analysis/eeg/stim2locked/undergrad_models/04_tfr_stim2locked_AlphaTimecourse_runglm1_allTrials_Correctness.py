# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:59:55 2024

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
glmdir = op.join(wd, 'glms', 'stim2locked', 'alpha_timecourses', 'undergrad_models', 'glm1')
if not op.exists(glmdir):
    os.mkdir(glmdir)

for i in subs:
    for iglm in [0, 1]: #controls whether glm is run on un-baselined or baselined data
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
            addtopath = '_baselined_pres2'
            baseline_input = True
            baseline_pres2 = True
        elif iglm == 1:
            addtopath = ''
            baseline_input = True
            baseline_pres2 = False
           
        posterior_channels = ['PO7', 'PO3', 'O1', 'O2', 'PO4', 'PO8', 'Oz', 'POz']
        
        #set baseline params to test stuff
        if baseline_input and not baseline_pres2:
            tfrdata = tfr.data.copy()
            for itime in range(tfr.times.size): #loop over timepoints
                tfrdata[:,:,:,itime] = np.subtract(tfrdata[:,:,:,itime], blines)
            tfr.data = tfrdata #re-assign the baselined data back into the tfr object
        elif baseline_input and baseline_pres2:
            bline = (-0.6, -0.3) #baseline to prior to stim2 onset
            tfr = tfr.apply_baseline(bline) #apply baseline
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
        
        trialnum = tfr.metadata.trlidz.to_numpy() #regressor for epoch number (proxy for time on task)
        acc      = tfr.metadata.PerceptDecCorrect.to_numpy()
        acc      = np.where(acc == 1, 1, -1)
        acc      = sp.stats.zscore(acc) #zscore this across trials

        DC = glm.design.DesignConfig()
        DC.add_regressor(name = 'intercept', rtype = 'Constant') #add intercept to model average lateralisation
        DC.add_regressor(name = 'correctness', rtype = 'Parametric', datainfo = 'correctness', preproc = None)
        DC.add_regressor(name = 'trialnumber', rtype = 'Parametric', datainfo = 'trialnum', preproc = None)
        DC.add_simple_contrasts() #add basic diagonal matrix for copes
        DC.add_contrast(values = [1, 1, 0], name = 'correct')
        DC.add_contrast(values = [1,-1, 0], name = 'incorrect')
    
        #create glmdata object
        glmdata = glm.data.TrialGLMData(data = tfrdat, time_dim = 1, sample_rate = 100,
                                        #add in metadata that's used to construct the design matrix
                                        correctness = acc,
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
        
        np.save(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_betas%s.npy'%addtopath), arr = betas)
        np.save(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_copes%s.npy'%addtopath), arr = copes)
        np.save(file = op.join(glmdir, param['subid'] + '_stim2lockedTFR_tstats%s.npy'%addtopath), arr = tstats)
        
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