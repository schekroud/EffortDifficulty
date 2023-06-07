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

sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
import eyefuncs as eyes

wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)

eyedir = op.join(wd, 'data', 'eyes')

subs = [3, 4, 5, 7, 9, 10, 11]

plot_steps = False
#%%
subdata = []
subcount = 0
for sub in subs:
    subcount +=1
    print('working on participant %01d/%1d'%(subcount, len(subs)))
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
            data = eyes.parse_eye_data(op.join(eyedir, 'asc', fname),
                                       block_rec = True, trial_rec = False, nblocks = beh_nblocks,
                                       binocular = False) #read in the data
            
            if not op.exists(rawname):
                with open(rawname, 'wb') as handle:
                    pickle.dump(data, handle)
        
        # if sub == 6:
        #     #remove last block of data
        #     data = data[:-1]
        
        nblocks = len(data)
        ds = deepcopy(data) #only used for plotting the raw, uninterpolated traces later
        #this will just nan any points that aren't possible with the screen dimensions you have (something went wrong, or tracker fritzed)
        print('\nreplacing datapoints found outside the screen dimensions')
        data = eyes.replace_points_outside_screen(data = data, nblocks = nblocks,
                                                  screen_dim = [1920, 1080],
                                                  adjust_pupils = False)
        
            
        #first, if they had a PLR routine then we need to cut this out of the data
        print('removing the PLR data')
        data = eyes.remove_before_task_start(data,
                                             snip_amount = 3, #seconds)
                                             srate = 1000)
        
        #and we will snip the end of each block to make it end shortly after the last trigger of the task
        data = eyes.snip_end_of_blocks(data, snip_amount = 2, srate = 1000)
        
        
        time = deepcopy(data[0]['trackertime']) - data[0]['trackertime'][0]
        #gets blinks using info from the pupil trace. sets blink periods to nan
        print('\nfinding blinks from the pupil trace')
        data_cleaned = eyes.cleanblinks_usingpupil(data, nblocks = nblocks)
        
        #finds all the missing periods (nans) in the data
        print('identifying nan periods in the data to be cleaned')
        data_cleaned = eyes.find_missing_periods(data_cleaned, nblocks = nblocks)
        
        #check in case there are missing data periods at the start of the recording as this messes up preproc a lot
        #after first finding missing data periods, it specifies these as 'blinks'
        #if a blink duration is negative, it means that an earlier (e.g. the first blink) didn't have a start, so it goes end of first  -> start of second
        #this next function checks for nans at the start of the recording (i.e. if it starts with missing data) which causes this
        print('checking if the recording starts with missing data')
        data_cleaned = eyes.check_nansatstart(data_cleaned)
        
        #finds all the missing periods (nans) in the data again after correcting the start of the recording where necessary
        print('identifying nan periods in the data to be cleaned')
        data_cleaned = eyes.find_missing_periods(data_cleaned, nblocks = nblocks)
        
        datcop = deepcopy(data_cleaned)
        blockid = 1
        
        
        interpolateBlinks = False
        if interpolateBlinks:
            if datcop[0]['binocular']:
                clean_traces = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry']
            elif not datcop[0]['binocular']: #monocular data
                clean_traces = ['x', 'y', 'p']
            print('\ninterpolating over nan periods')
            for block in range(len(data_cleaned)):
                print('cleaning data for block %02d/%02d'%(block+1,nblocks))
                for trace in clean_traces:
                    data_cleaned[block] = eyes.interpolateBlinks_Blocked(data_cleaned[block],trace)
                    #there is a problem with this function - some of the blinks end the signal on a nan value so it interpolates nans and doesn't clean the signal at all.
                    #its off by one sample for some reason (usually)
                blockid +=1
            
            
        data_cleaned = eyes.transform_pupil(data_cleaned)    
        
    subdata.append(data_cleaned)
    
    for block in range(len(data_cleaned)):
        if data_cleaned[0]['binocular']:
            print(np.isnan(data_cleaned[block]['lp']).sum())
        elif not data_cleaned[0]['binocular']:
            print(np.isnan(data_cleaned[block]['p']).sum())
    
    #check quality of blink removal
    # blockid = 3
    # tmpplot = deepcopy(data_cleaned)[blockid]['p']
    # lpmin = np.nanmin(data_cleaned[blockid]['p'])
    # lpmax = np.nanmax(data_cleaned[blockid]['p'])
    # tmpblinks = deepcopy(data_cleaned[blockid])['Eblk_p']
    # missingstarts = np.array(tmpblinks)[:,1].astype(int).tolist()
    # missingends   = np.array(tmpblinks)[:,2].astype(int).tolist()
    # plt.figure()
    # plt.plot(ds[blockid]['p'], color = '#bdbdbd', label = 'raw data')
    # plt.plot(tmpplot, color = '#696969', label = 'cleaned data')
    # plt.vlines(x = missingstarts, ymin = lpmin, ymax = lpmax, ls = 'dashed', color = '#6baed6', lw = 1)
    # plt.vlines(x = missingends, ymin = lpmin, ymax = lpmax, ls = 'dashed', color = '#f16913', lw = 1)
    # plt.vlines(x = 919963, ymin = lpmin, ymax = lpmax, ls = 'dashed', color = '#000000', lw = 2)
    # plt.title('subject %02d pupil, check preproc quality'%sub)
    # plt.legend(loc='lower right')
    
    # tmpplot = deepcopy(data_cleaned)[blockid]['x']
    # lpmin = np.nanmin(data_cleaned[blockid]['x'])
    # lpmax = np.nanmax(data_cleaned[blockid]['x'])
    # tmpblinks = deepcopy(data_cleaned[blockid])['Eblk_x']
    # missingstarts = np.array(tmpblinks)[:,1].astype(int).tolist()
    # missingends   = np.array(tmpblinks)[:,2].astype(int).tolist()
    # plt.figure()
    # plt.plot(ds[blockid]['x'], color = '#bdbdbd', label = 'raw data')
    # plt.plot(tmpplot, color = '#696969', label = 'cleaned data')
    # # plt.vlines(x = missingstarts, ymin = 0, ymax = 1920, ls = 'dashed', color = '#6baed6', lw = 0.5)
    # # plt.vlines(x = missingends, ymin = 0, ymax = 1920, ls = 'dashed', color = '#f16913', lw = 0.5)
    # plt.title('subject %02d x, check preproc quality'%sub)
    
    if not op.exists(savename):
        print('\nsaving the preprocessed data to file')
        with open(savename, 'wb') as handle:
            pickle.dump(data_cleaned, handle)
#%%
            