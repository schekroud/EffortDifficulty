#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:34:58 2023

@author: sammi
"""

import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import pickle
import sklearn as skl
from sklearn import *
%matplotlib

sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
import eyefuncs as eyes

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)

eyedir = op.join(wd, 'data', 'eyes')

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,             39])
#29/30 just need some fixing
#36/37 are unusable
#38 we changed the eye that was tracked partway through, so it is struggling to read this dataset properly (as it expects a specific channel name)
#can probably patch this, but ignore for now

plot_steps = False
#%%
subdata = []
subcount = 0
for sub in subs:
    subcount +=1
    print('\n- - - - - - working on participant %s (%01d/%1d) - - - - - -\n'%(str(sub), subcount, len(subs)))
    fname = 'EffDS%02da.asc'%sub #get the asc file name
    savename = 'EffDS%02da_preproc.pickle'%sub
    savename = op.join(eyedir, 'preprocessed',  savename)
    behfname = op.join(wd, 'data', 'datafiles', 'EffortDifficulty_s%02d_combined.csv'%sub)
    
    #load in behavioural data
    bdata = pd.read_csv(behfname)
    beh_nblocks = len(bdata.blocknumber.unique())
    
    if op.exists(savename):
        print('preprocessed data exists - reading it in')
        with open(savename, 'rb') as handle:
            data_cleaned = pickle.load(handle)
    else:
        rawname = 'EffDS%02da_raw.pickle'%sub
        rawname = op.join(eyedir, 'raw', rawname)
        
        if op.exists(rawname):
            print('raw data object exists, reading it in')
            with open(rawname, 'rb') as handle:
                data = pickle.load(handle)
        else:
            if sub == 6:
                beh_nblocks = 4
                #this subject did 3 blocks of task,
                #but you can only quit at the start of a block so it still got a start/end message
            print('\nloading in participant data')
            data = eyes.parse_eye_data(eye_fname = op.join(eyedir, 'asc', fname),
                                       block_rec = True,
                                       trial_rec = False,
                                       nblocks = beh_nblocks,
                                       binocular = False) #read in the data
            
            if not op.exists(rawname):
                with open(rawname, 'wb') as handle:
                    pickle.dump(data, handle)
            
            
        #first, if they had a PLR routine then we need to cut this out of the data as it can cause issues down the line
        print('removing the PLR data')
        data = eyes.remove_before_task_start(data,
                                             snip_amount = 4, #seconds)
                                             srate = 1000)
        
        #this removes eyelink messages that we aren't going to use. keeps data size smaller and less clunky
        data = eyes.drop_eyelink_messages(data)
        ds = deepcopy(data) #useful for plotting raw data traces to check preprocessing effects
        
        #and we will snip the end of each block to make it end shortly after the last trigger of the task
        data = eyes.snip_end_of_blocks(data, snip_amount = 3, srate = 1000)

        
        #plot each block of pupil stacked so we can just look
        # fig = plt.figure()
        # for iblock in range(len(data)):
        #     ax = fig.add_subplot(len(data), 1, iblock+1)
        #     ax.plot(data[iblock]['trackertime'], data[iblock]['p'], lw = 1, color = '#3182bd')
        #     ax.plot(data[iblock]['trackertime'], np.where(data[iblock]['p'] == 0, np.nan, data[iblock]['p']), color = '#31a354', lw=1)
        #     ax.set_title('block '+str(iblock+1))
        
        
        #this sets periods where the pupil trace is zero (missing data) to nan (making it properly missing)
        # data = eyes.nan_missingdata(data)
        
        #next step is to nan the period around blinks as this is usually noisy and contaminated
        
        
        #and we will snip the end of each block to make it end shortly after the last trigger of the task
        
        
        time = deepcopy(data[0]['trackertime']) - data[0]['trackertime'][0]
        #gets blinks using info from the pupil trace. sets blink periods to nan
        print('\nfinding blinks from the pupil trace')
        data_cleaned = eyes.cleanblinks_usingpupil(data, nblocks = beh_nblocks)
        
        #visualise the effect of this
        
        fig = plt.figure()
        for iblock in range(len(data)):
            tmpdata = deepcopy(data_cleaned[iblock])
            tmpp = deepcopy(tmpdata['p'])
            tmpp[tmpdata['badsamps']] = np.nan
            ax = fig.add_subplot(len(data), 1, iblock+1)
            ax.plot(ds[iblock]['trackertime'], ds[iblock]['p'], lw = 1, color = '#3182bd', label = 'pupil')
            ax.plot(tmpdata['trackertime'], tmpp, lw = 1, color = '#fdae6b')
        
        #data_cleaned has bad samples as nans and no other information stored. can probably pass this on for future stuff.
        
        data_t = eyes.transform_pupil(data_cleaned)
        
        fig = plt.figure()
        for iblock in range(len(data)):
            ax = fig.add_subplot(len(data), 1, iblock+1)
            # ax.plot(ds[iblock]['trackertime'], ds[iblock]['p'], lw = 1, color = '#3182bd', label = 'pupil')
            ax.plot(data_t[iblock]['trackertime'], data_t[iblock]['p_perc'], lw = 1, color = '#fdae6b')
        
        #save this data (nan periods are left as is and not interpolated)
        
        if not op.exists(savename):
            print('\nsaving the preprocessed data to file')
            with open(savename, 'wb') as handle:
                pickle.dump(data_cleaned, handle)
    