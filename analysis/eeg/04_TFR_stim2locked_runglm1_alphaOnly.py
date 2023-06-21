#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:24:23 2023

@author: sammi
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

# sys.path.insert(0, '/ohba/pi/knobre/schekroud/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo

# wd = '/ohba/pi/knobre/schekroud/postdoc/student_projects/EffortDifficulty' #workstation wd
wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
os.chdir(wd)

# sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm


subs = np.array([10, 11, 12, 13, 14, 15, 16])
#%%
glms2run = 1 #1 with no baseline, one where tfr input data is baselined
for i in subs:
    for iglm in [1]: #force this to be baselined for now #range(glms2run):
        print('\nworking on subject ' + str(i) +'\n')
        sub   = dict(loc = 'laptop', id = i)
        param = getSubjectInfo(sub)
        tfr = mne.time_frequency.read_tfrs(param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned_Alpha').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
        
        #comes with metadata attached            
        tfr = tfr['fbtrig != 62'] #drop timeout trials
    
        # tmpleft  = deepcopy(tfr)['chosenside==0'].average()
        # tmpright = deepcopy(tfr)['chosenside==1'].average()
        # tmp_lvsr = tmpleft - tmpright
    
        if iglm == 0:
            addtopath = ''
            baseline_input = False
        elif iglm == 1:
            addtopath = '_baselined'
            baseline_input = True

        if baseline_input:
           print(' -- baselining the TFR data -- ')
           tfr = tfr.apply_baseline((-0.15, 0.15)) #this takes 300ms around the onset so it might work for pre-stim stuff
    
        glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)
        nobs = glmdata.num_observations
        trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
    
        # if iglm == 0:
        #     print('\nSubject %02d has %03d trials\n'%(i, trials.size))
        correctness = tfr.metadata.PerceptDecCorrect.to_numpy()
        correct     = tfr.metadata.rewarded.to_numpy()
        incorrect   = tfr.metadata.unrewarded.to_numpy()
        correctness = np.where(correctness == 0, -1, correctness)
        
        
        regressors = list()

        #add regressors to the model
        regressors.append(glm.regressors.CategoricalRegressor(category_list = trials, codes = 1, name = 'grandmean'))
        regressors.append(glm.regressors.ParametricRegressor(name = 'correctness', values = correctness, preproc = None, num_observations = nobs)) #contrast regressor
        
        
        contrasts = list()
        contrasts.append(glm.design.Contrast([1, 0], 'grandmean'))      #0
        contrasts.append(glm.design.Contrast([0, 1], 'correctness'))    #1
        contrasts.append(glm.design.Contrast([1, 1], 'correct'))        #2
        contrasts.append(glm.design.Contrast([1,-1], 'incorrect'))      #3
        
        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        #glmdes.plot_summary()
            #if iglm == 0:
            #    glmdes.plot_summary()
    
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info

        del(tfr)
    
    
        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel( glmdes, glmdata)

        del(glmdata) #clear from RAM as not used from now on really
#        contrastnames = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)

        # loop over betas first and save them
        for iname in range(len(model.regressor_names)): #loop over regressors
            if model.regressor_names[iname] in ['grandmean', 'correctness']:
                nave = len(trials)
            elif model.regressor_names[iname] == 'correct':
                nave = correct.sum()
            elif model.regressor_names[iname] == 'incorrect':
                nave = incorrect.sum()
        
            ibeta_name = model.regressor_names[iname] #regressor name to save
            
            tfr_beta = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                     data = model.betas[iname])
            tfr_beta.save(fname = op.join(wd, 'glms', 'stim2locked', 'glm1', 
                                          param['subid'] + '_stim2locked_tfr_AlphaOnly_' + ibeta_name + '_beta-tfr.h5'), overwrite = True)
            del(tfr_beta)

        #loop over contrasts + tstats now
        for iname in range(len(model.contrast_names)):
            name = model.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name

            if iname in [2]:
                nave = correct.sum()
            elif iname in [3]:
                nave = incorrect.sum()
            else:
                nave = len(trials)

            #save betas, contrasts and tstats
            # betas[count,:,:] = model.betas
            # copes[count,:,:] = model.copes
            # tstats[count,:,:] = model.get_tstats()
            

            tfr_cope = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = model.copes[iname])
            
            tfr_cope.save(fname = op.join(wd, 'glms', 'stim2locked', 'glm1', 
                                          param['subid'] + '_stim2locked_tfr_AlphaOnly_' + name + '_cope-tfr.h5'), overwrite = True)
            del(tfr_cope)

            tfr_tstat = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = model.get_tstats()[iname])
            tfr_tstat.save(fname = op.join(wd, 'glms', 'stim2locked', 'glm1', 
                                          param['subid'] + '_stim2locked_tfr_AlphaOnly_' + name + '_tstat-tfr.h5'), overwrite = True)
            del(tfr_tstat)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)