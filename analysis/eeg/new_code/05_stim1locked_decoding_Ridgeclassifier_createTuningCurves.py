# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:25:26 2024

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import sklearn as skl
from sklearn import *
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
%matplotlib

loc = 'laptop'
if loc == 'laptop':
    sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
elif loc == 'workstation':
    sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

from funcs import getSubjectInfo
from decoding_functions import run_decoding, run_decoding_predproba
from joblib import parallel_backend

os.chdir(wd)
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

figpath = op.join(wd, 'figures', 'decoding', 'LDA')

import progressbar
progressbar.streams.flush()

nsamples = 200 #samples in the epoch at 100hz sample rate
# nsamples = 1001 #samples in the epoch at 500hz sample rate
accuracies = np.zeros(shape = [subs.size, nsamples])
diffaccuracies = np.zeros(shape = [subs.size, 4, nsamples])
use_vischans_only  = True
use_pca            = False #using pca after selecting only visual channels is really bad for the decoder apparently
smooth_singletrial = False
testtype           = 'rskf'
# testtype = 'leaveoneout'

nbins      = 8
centprobs  = np.zeros(shape = [subs.size, 2, nbins-1, nsamples]) #store the average tuning curve for each participant
centprobs2 = np.zeros(shape = [subs.size, 2, nbins-1, nsamples]) #store the average tuning curve for each participant

progressbar.streams.wrap_stderr()
bar = progressbar.ProgressBar(max_value = len(subs)).start()
subcount = -1
for i in subs:
    subcount += 1
    bar.update(subcount+1)
    # sub   = dict(loc = 'laptop', id = i)
    sub   = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    print(f'\n- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'),
                                verbose = 'ERROR', preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -0.5, tmax = 1.5) #crop for speed and relevance  
    epochs.resample(100)
    times = epochs.times
    
    if i == 16:
        epochs = epochs['blocknumber != 4']
    if i in [21, 25]:
        #these participants had problems in block 1
        epochs = epochs['blocknumber > 1']
    
    if use_vischans_only:
        #drop all but the posterior visual channels
        epochs = epochs.pick(picks = [
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                            'O1', 'Oz', 'O2'])
    data   = epochs._data.copy() #get the data matrix
    if smooth_singletrial:
        data = sp.ndimage.gaussian_filter1d(data.copy(), sigma = 7) #this is equivalent to smoothing with a 14ms gaussian blur as 500Hz sample rate.
        #this should help with some denoising
    bdata  = epochs.metadata.copy() 
    [ntrials, nfeatures, ntimes] = data.shape #get data shape -- [trials x channels x timepoints]

    orientations = bdata.stim1ori.to_numpy()
    
    convprobs     = np.empty(shape = [2, nbins-1, epochs.times.size])
    nonconvprobs  = np.zeros_like(convprobs) * np.nan
    nonconvprobs2 = np.zeros_like(convprobs) * np.nan
    
    oricount = -1
    for oritype in ['neg', 'pos']:
        oricount += 1
        if oritype == 'pos':
            useoris = np.greater(orientations, 0)
        elif oritype == 'neg':
            useoris = np.less(orientations, 0)
        
        oris = orientations[useoris]
        bins = np.linspace(oris.min(), oris.max()+1, nbins)
        labelbins = np.digitize(oris, bins)
        nclasses = np.unique(labelbins).size
        
        data_use = data.copy()[useoris, :] #take just the positively oriented data to begin with
        
        
        #pre-create some variables to store info
        label_pred = np.zeros(shape = [len(oris), ntimes]) * np.nan #store the predicted label for each trial and time
        accuracy = np.zeros(ntimes) #store across-trial classifier accuracy at each time point
        
        predictedprobs     = np.zeros(shape = [len(oris), nclasses, ntimes]) #store the P(class) for each class, per trial and time point
        centred_probabilities = np.zeros_like(predictedprobs) #store P(class), centred on the predicted class
        centred_probabilities2 = np.zeros_like(predictedprobs) #store P(class), centred on the predicted class
        
        for tp in range(ntimes): #for each time point
            preds, predprobas = run_decoding_predproba(data_use, labelbins, tp, use_pca, testtype, classifier='svm') #run decoding
            #this outputs:
            # preds -- the predicted class for each trial (ntrials)
            # predprobas - the P(class) for each class (ntrials x nclasses)
            
            label_pred[:,tp] = preds.astype(int) #store the across-trial predicted class labels for this time point
            
            #we want to shift these predicted probabilities, centering them on the predicted class label
            predprobas_shifted  = np.zeros_like(predprobas.copy()) * np.nan #create a new, nan-array to store the centred class probabilities
            predprobas_shifted2 = np.zeros_like(predprobas.copy()) * np.nan #create a new, nan-array to store the centred class probabilities
            
            #need to shift predprobas to centre on the predicted class
            for rowid in range(0, predprobas.shape[0]): #loop over rows(trials) of the class prediction probabilities
                ilabel = preds[rowid] #get the trial
                ilabel2 = labelbins[rowid] #get the actual class on this trial rather than predicted class
                # labind = ilabel-1 #the label isnt in python indexing, so lets just fix this
                nshift = int(np.floor((nclasses+1)/2)-ilabel)
                nshift2 = int(np.floor((nclasses+1)/2)-ilabel2)
                shifted = np.roll(predprobas[rowid,:], nshift) #centre the class predictions on the predicted class
                shifted2 = np.roll(predprobas[rowid,:], nshift2)
                if nshift >0:
                    shifted[:nshift] = np.nan #set the shifted vals to nan
                elif nshift < 0:
                    shifted[nshift:] = np.nan
                
                if nshift2 >0:
                    shifted2[:nshift2] = np.nan #set the shifted vals to nan
                elif nshift2 < 0:
                    shifted2[nshift2:] = np.nan
                
                predprobas_shifted[rowid] = shifted.copy()
                predprobas_shifted2[rowid] = shifted2.copy()
            
            
            
            centred_probabilities[:,:,tp]  = predprobas_shifted #store the label centred predicted probabilities
            centred_probabilities2[:,:,tp] = predprobas_shifted2 #store the label centred predicted probabilities
            predictedprobs[:,:,tp]         = predprobas         #store the class predicted probabilities
        
        nonconvprobs[oricount]  = np.nanmean(centred_probabilities, axis=0)
        nonconvprobs2[oricount] = np.nanmean(centred_probabilities2, axis=0)
    
    centprobs[subcount] = nonconvprobs #store the across trial centred prediction probabilities
    centprobs2[subcount] = nonconvprobs2 #store the across trial centred prediction probabilities

#%%

#centprobs is centred on the predicted class
#centprobs2 is centred on the actual orientation 

gmean_centprobs = centprobs.copy().mean(0)

fig = plt.figure(figsize = [12, 6])
ax = fig.add_subplot(121)
plot = ax.imshow(gmean_centprobs[0].T, 
      aspect='auto', origin = 'lower', #extent = np.array([times.min(), times.max(), 0, nclasses]), #vmin = vmin, vmax = vmax,
      extent = [0, nclasses, times.min(), times.max()],
      cmap = 'RdBu_r', interpolation = 'none')#, vmin = 1/nclasses - 0.05, vmax = 1/nclasses +0.05 )
ax.set_xticks(np.arange(0.5, nclasses+.5), labels = np.arange(1, nclasses+1).astype(str))
ax.set_xlabel('orientation class')
ax.set_ylabel('time rel. to stim1 onset (s)')
ax.set_ylim([0, 1])
ax.set_title('anti clockwise')
# fig.colorbar(plot)

ax = fig.add_subplot(122)
plot = ax.imshow(gmean_centprobs[1].T, 
      aspect='auto', origin = 'lower', #extent = [times.min(), times.max(), 0, nclasses], #vmin = vmin, vmax = vmax,
      extent = [0, nclasses, times.min(), times.max()],
      cmap = 'RdBu_r', interpolation = 'none')#, vmin = 1/nclasses - 0.05, vmax = 1/nclasses +0.05 )
ax.set_xticks(np.arange(0.5, nclasses+.5), labels = np.arange(1, nclasses+1).astype(str))
ax.set_xlabel('orientation class')
ax.set_ylabel('time rel. to stim1 onset (s)')
ax.set_ylim([0, 1])
ax.set_title('clockwise')
fig.colorbar(plot)
# 

negoris = np.arange(-60, -30+1)
posoris = np.arange(30, 60+1)

negbins = np.linspace(negoris.min(), negoris.max()+1, nbins)
posbins = np.linspace(posoris.min(), posoris.max()+1, nbins)

negbinmids = (negbins[1:] + negbins[:-1])/2 + 45
posbinmids = (posbins[1:] + posbins[:-1])/2 - 45

negtheta = np.cos(np.radians(negbinmids))
postheta = np.cos(np.radians(posbinmids))


convedprobs = np.empty(centprobs.shape) * np.nan

for isub in range(subs.size): #loop overtrials
   distances = centprobs.copy()[isub]
   negdists  = distances.copy()[0]
   posdists  = distances.copy()[1]
   
   for tp in range(ntimes):
       tneg = negtheta * negdists[:,tp]
       tpos = postheta * posdists[:,tp]
       
       convedprobs[isub, 0, :, tp] = tneg
       convedprobs[isub, 1, :, tp] = tpos

gmean_convedprobs = np.nanmean(convedprobs, axis=0)#.mean(axis=0)

fig = plt.figure(figsize = [12, 6])
ax = fig.add_subplot(121)
plot = ax.imshow(gmean_convedprobs[0].T, #gmean_convedprobs.T, #
      aspect='auto', origin = 'lower', #extent = np.array([times.min(), times.max(), 0, nclasses]), #vmin = vmin, vmax = vmax,
      extent = [0, nclasses, times.min(), times.max()],
      cmap = 'RdBu_r', interpolation = 'none')#, vmin = 1/nclasses - 0.05, vmax = 1/nclasses +0.05 )
ax.set_xticks(np.arange(0.5, nclasses+.5), labels = np.arange(1, nclasses+1).astype(str))
ax.set_xlabel('orientation class')
ax.set_ylabel('time rel. to stim1 onset (s)')
ax.set_ylim([0, 1])
ax.set_title('anti clockwise')
fig.colorbar(plot)

ax = fig.add_subplot(122)
plot = ax.imshow(gmean_convedprobs[1].T, 
      aspect='auto', origin = 'lower', #extent = [times.min(), times.max(), 0, nclasses], #vmin = vmin, vmax = vmax,
      extent = [0, nclasses, times.min(), times.max()],
      cmap = 'RdBu_r', interpolation = 'none')#, vmin = 1/nclasses - 0.05, vmax = 1/nclasses +0.05 )
ax.set_xticks(np.arange(0.5, nclasses+.5), labels = np.arange(1, nclasses+1).astype(str))
ax.set_xlabel('orientation class')
ax.set_ylabel('time rel. to stim1 onset (s)')
ax.set_ylim([0, 1])
ax.set_title('clockwise')
fig.colorbar(plot)
