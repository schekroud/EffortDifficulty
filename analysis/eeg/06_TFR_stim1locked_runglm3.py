# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:31:26 2023

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
import funcs
from funcs import getSubjectInfo, gesd, plot_AR

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)
import glmtools as glm


subs = np.array([10, 11, 12, 13, 14, 15, 16])
#%%
glms2run = 1 #1 with no baseline, one where tfr input data is baselined
for i in subs:
    for iglm in [0,1]: #controls whether glm is run on un-baselined or baselined data
        print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
        sub   = dict(loc = 'workstation', id = i)
        param = getSubjectInfo(sub)
        tfr = mne.time_frequency.read_tfrs(param['stim1locked'].replace('stim1locked', 'stim1locked_cleaned').replace('-epo.fif', '-tfr.h5')); tfr = tfr[0]
        
        #comes with metadata attached            
        tfr = tfr['fbtrig != 62'] #drop timeout trials
        
        #need to get difficulty streaks to remove the first few trials when difficulty has changed
        #read full behavioural data to do this, then subset it to the cleaned trials
        #(effort, can fix later by doing this when making the raw dataframes)
        
        bdata = pd.read_csv(param['behaviour'])
        bdata = funcs.get_difficulty_sequences(bdata)
        
        
            
        if iglm == 0:
            addtopath = ''
            baseline_input = False
        elif iglm == 1:
            addtopath = '_baselined'
            baseline_input = True

        if baseline_input:
           print(' -- baselining the TFR data -- ')
           tfr = tfr.apply_baseline((-2.7, -2.4)) #takes 1 to 0.8s before stim1 onset as baseline, shouldn't interfere with prestim effects here
    
        glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)
        nobs = glmdata.num_observations
        trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
    
        # if iglm == 0:
        #     print('\nSubject %02d has %03d trials\n'%(i, trials.size))
        correctness = tfr.metadata.PerceptDecCorrect.to_numpy()
        correct     = tfr.metadata.rewarded.to_numpy()
        incorrect   = tfr.metadata.unrewarded.to_numpy()
        correctness = np.where(correctness == 0, -1, correctness)
        difficultyOri = tfr.metadata.difficultyOri.to_numpy()
        difficulty = np.subtract(difficultyOri, difficultyOri.mean()) #demean, so this is relative to the average difficulty across the task
        
        
        d2  = np.where(difficultyOri == 2, 1, 0)
        d4  = np.where(difficultyOri == 4, 1, 0)
        d8  = np.where(difficultyOri == 8, 1, 0)
        d12 = np.where(difficultyOri == 12, 1, 0)

        
        
        regressors = list()

        #add regressors to the model
        regressors.append(glm.regressors.CategoricalRegressor(category_list = d2, codes = 1, name = 'diff2'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = d4, codes = 1, name = 'diff4'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = d8, codes = 1, name = 'diff8'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = d12, codes = 1, name = 'diff12'))
        regressors.append(glm.regressors.ParametricRegressor(name = 'correctness', values = correctness, preproc = None, num_observations = nobs)) #contrast regressor
        
        
        contrasts = list()
        
        contrasts.append(glm.design.Contrast([1, 0, 0, 0, 0], 'diff2'))
        contrasts.append(glm.design.Contrast([0, 1, 0, 0, 0], 'diff4'))
        contrasts.append(glm.design.Contrast([0, 0, 1, 0, 0], 'diff8'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 1, 0], 'diff12'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 0, 1], 'correctness'))
        contrasts.append(glm.design.Contrast([1, 1, 1, 1, 0], 'grandmean'))
                  
        
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
            if model.regressor_names[iname] in ['grandmean', 'correctness', 'difficulty']:
                nave = len(trials)
            elif model.regressor_names[iname] == 'diff2':
                nave = d2.sum()
            elif model.regressor_names[iname] == 'diff4':
                nave = d4.sum()
            elif model.regressor_names[iname] == 'diff8':
                nave = d8.sum()
            elif model.regressor_names[iname] == 'diff12':
                nave = d12.sum()
            
        
            ibeta_name = model.regressor_names[iname] #regressor name to save
            
            tfr_beta = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                     data = model.betas[iname])
            tfr_beta.save(fname = op.join(wd, 'glms', 'stim1locked', 'glm3', 
                                          param['subid'] + '_stim1locked_tfr_' + ibeta_name + addtopath + '_beta-tfr.h5'), overwrite = True)
            del(tfr_beta)

        #loop over contrasts + tstats now
        for iname in range(len(model.contrast_names)):
            name = model.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name

            if iname in [0]:
                nave = d2.sum();
            elif iname in [1]:
                nave = d4.sum();
            elif iname in [2]:
                nave = d8.sum();
            elif iname in [3]:
                nave = d12.sum();
            else:
                nave = len(trials)

            #save betas, contrasts and tstats
            # betas[count,:,:] = model.betas
            # copes[count,:,:] = model.copes
            # tstats[count,:,:] = model.get_tstats()
            

            tfr_cope = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = model.copes[iname])
            
            tfr_cope.save(fname = op.join(wd, 'glms', 'stim1locked', 'glm3', 
                                          param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_cope-tfr.h5'), overwrite = True)
            del(tfr_cope)

            tfr_tstat = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = model.get_tstats()[iname])
            tfr_tstat.save(fname = op.join(wd, 'glms', 'stim1locked', 'glm3', 
                                          param['subid'] + '_stim1locked_tfr_' + name + addtopath + '_tstat-tfr.h5'), overwrite = True)
            del(tfr_tstat)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)