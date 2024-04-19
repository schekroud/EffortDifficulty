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

loc = 'workstation'
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
nsamples = 1001 #samples in the epoch at 500hz sample rate
accuracies = np.zeros(shape = [subs.size, nsamples])
diffaccuracies = np.zeros(shape = [subs.size, 4, nsamples])
use_vischans_only  = True
use_pca            = False #using pca after selecting only visual channels is really bad for the decoder apparently
smooth_singletrial = False
testtype           = 'rskf'

nbins = 8
centprobs = np.zeros(shape = [subs.size, nbins-1, nsamples]) #store the average tuning curve for each participant

progressbar.streams.wrap_stderr()
bar = progressbar.ProgressBar(max_value = len(subs)).start()
subcount = -1
for i in subs:
    subcount += 1
    bar.update(subcount+1)
    # sub   = dict(loc = 'laptop', id = i)
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    print(f'\n- - - - working on subject {i} - - - -')
    
    epochs    = mne.read_epochs(param['stim1locked'].replace('stim1locked', 'stim1locked_icaepoched_resamp_cleaned'),
                                verbose = 'ERROR', preload=True) #already baselined
    dropchans = ['RM', 'LM', 'EOGL', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -0.5, tmax = 1.5) #crop for speed and relevance  
    
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
    
    
    posoris = orientations[orientations>0] #take just the positive orientations here
    bins = np.linspace(30, 60+1, nbins)
    labelbins = np.digitize(posoris, bins)
    #pd.value_counts(labelbins)
    nclasses = np.unique(labelbins).size
    
    data_pos = data.copy()[orientations>0, :] #take just the positively oriented data to begin with
    
    
    #pre-create some variables to store info
    label_pred = np.zeros(shape = [len(posoris), ntimes]) * np.nan #store the predicted label for each trial and time
    accuracy = np.zeros(ntimes) #store across-trial classifier accuracy at each time point
    
    predictedprobs     = np.zeros(shape = [len(posoris), nclasses, ntimes]) #store the P(class) for each class, per trial and time point
    centred_probabilities = np.zeros_like(predictedprobs) #store P(class), centred on the predicted class
    
    for tp in range(ntimes): #for each time point
        preds, predprobas = run_decoding_predproba(data_pos, labelbins, tp, use_pca, testtype, classifier='ridge') #run decoding
        #this outputs:
        # preds -- the predicted class for each trial (ntrials)
        # predprobas - the P(class) for each class (ntrials x nclasses)
        
        label_pred[:,tp] = preds.astype(int) #store the across-trial predicted class labels for this time point
        
        #we want to shift these predicted probabilities, centering them on the predicted class label
        predprobas_shifted = np.zeros_like(predprobas) * np.nan #create a new, nan-array to store the centred class probabilities
        
        #need to shift predprobas to centre on the predicted class
        for rowid in range(0, predprobas.shape[0]): #loop over rows(trials) of the class prediction probabilities
            ilabel = preds[rowid] #get the trial
            predprobas_shifted[rowid] = np.roll(predprobas[rowid,:], int(np.floor(nbins/2) - ilabel)) #centre the class predictions on the predicted class
        
        centred_probabilities[:,:,tp] = predprobas_shifted #store the label centred predicted probabilities
        predictedprobs[:,:,tp]        = predprobas         #store the class predicted probabilities
    
    #get across trial accuracy and store this
    # for tp in range(ntimes):
        
    
    smooth_probs = False
    if smooth_probs:
        raw_centred_probs = centred_probabilities.copy()
        centred_probabilities = sp.ndimage.gaussian_filter1d(centred_probabilities, sigma = 5)
    
    #to convolve with a cosine we need to get cos(theta) for the angles of the bins
    # angrange = 30 #bins between 30 and 60 degrees
    binmids = (bins[1:] + bins[:-1])/2 - 45 #get the middle of each bin, centre onf 45 (middle of the angle range)
    theta = np.cos(np.radians(binmids))
    # plt.figure(); plt.plot(binmids, theta)
    
    #do full range from -pi to pi
    # theta = np.arange((nclasses))/(nclasses)*(2*np.pi)-np.pi
    # plt.figure(); plt.plot(binmids, np.cos(theta))
    # theta = np.cos(theta)
    
    convolved_probas = np.zeros_like(centred_probabilities) * np.nan
   
    for trl in range(posoris.size): #loop overtrials
       distances = centred_probabilities.copy()[trl]
       convdist = np.zeros_like(distances) * np.nan
       for tp in range(ntimes):
           t = theta * distances[:,tp]
           convolved_probas[trl,:,tp] = t

    centprobs[subcount] = centred_probabilities.mean(axis=0) #store the across trial 'tuning curve'

#%%

gmean_centprobs = centprobs.copy().mean(0)
#for vis, add a new column to make it symmetrical
# gmean_centprobs = np.append(gmean_centprobs, gmean_centprobs[0,:].reshape(1,-1), axis=0)

fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
plot = ax.imshow(gmean_centprobs, aspect='auto', interpolation = 'none',
          extent = [epochs.times.min(), epochs.times.max(), 0, 6], vmin = -0.5, vmax = 0.5,
          cmap = 'RdBu_r')
fig.colorbar(plot)


conv = np.zeros_like(gmean_centprobs) * np.nan
theta2 = np.arange((nclasses))/(nclasses)*(np.pi)-np.pi/2
theta2 = np.cos(theta2)

for tp in range(nsamples):
    conv[:,tp] = theta2 * (gmean_centprobs[:,tp])#- (1/nclasses))

fig = plt.figure(figsize = [6,4])
ax = fig.add_subplot(111)
plot = ax.imshow(conv, aspect='auto', interpolation = 'none', vmin = -0.1, vmax =0.4,
          extent = [epochs.times.min(), epochs.times.max(), -15, 15],
          cmap = 'RdBu_r')
fig.colorbar(plot)

