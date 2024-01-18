#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:41:28 2024

@author: sammichekroud
"""

import pandas as pd
import numpy as np
import os
import os.path as op

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)
datapath = op.join(wd, 'data', 'datafiles')

sublist = np.arange(3, 40)

def streaks_numbers(array):
    '''
    finds streaks of an array where a number is the same
    - useful for finding sequences of trials where difficulty is the same
    note: bit slow because it literally loops over an array so it isn't the fastest, but it works
    '''
    x = np.zeros(array.size).astype(int)
    count = 0
    for ind in range(len(x)):
        if array[ind] == array[ind-1]:
            count += 1 #continuing the sequence
        else: #changed difficulty
            count = 1
        x[ind] = count
    return x

for isub in sublist:
    dfiles = sorted(os.listdir(op.join(datapath, 's%02d'%isub))) #get filenames for all behavioural data for the subject
    
    if isub == 29:
        #EEG crashed in this subject at the start of block 3. they completed a first session of two complete blocks, and second session after crash with 3 blocks
        #third block of the first session needs removing,
        #then need to recreate the 'blocknumber' column of the dataframe so no duplicates, to reflect what the ppt actually did
        dfiles = [x for x in dfiles if 's29_block_03.csv' not in x]
    
    
    dframes = [pd.read_csv(op.join(datapath, 's%02d'%isub, fname)) for fname in dfiles] #get all files in the directory as dataframes
    
        
    
    tmpdf = pd.concat(dframes) #bind blockwise data into one dataframe with all data
    if isub == 29:
        #recreate the blocknumber columna accordingly
        tmpdf = tmpdf.assign(blocknumber = np.sort(np.tile(np.arange(1, 6), int(len(tmpdf)/5))))
        
    tmpdf.insert(0, 'subid', isub) #add subject id into the dataframe
    #get label for feedback on each trial
    tmpdf = tmpdf.assign(
        fbgiven = np.select([np.equal(tmpdf.fbtrig, 60),
                             np.equal(tmpdf.fbtrig, 61),
                             np.equal(tmpdf.fbtrig, 62)],
                            ['correct', 'incorrect', 'timed out'], default = 'nan'))
    tmpdf = tmpdf.assign(prevtrlfb = tmpdf.fbgiven.shift(1)) #get previous trials feedback
    #get whether the trial was rewarded (correct) or unrewarded (incorrect) - same as fbgiven, but just as an integer really (and sets time out trials as 0)
    tmpdf = tmpdf.assign(rewarded = np.where(tmpdf.fbgiven == 'correct', 1, 0),
                         unrewarded = np.where(tmpdf.fbgiven == 'incorrect', 1, 0))
    
    bdat = pd.DataFrame()
    for iblock in tmpdf.blocknumber.unique():
        #this needs to be run separately for each block 
        tmpdata = tmpdf.query('blocknumber == @iblock')
        #add in where this trial is within a perceptual difficulty sequence
        tmpdata = tmpdata.assign(diffseqpos = streaks_numbers(tmpdata.difficultyOri.to_numpy()))
        tmpdata = tmpdata.assign(untilDiffSwitch = np.multiply(-1,np.flip(streaks_numbers(np.flip(tmpdata.difficultyOri.to_numpy())))))
        #reconstruct the behavioural data file
        bdat = pd.concat([bdat, tmpdata])
    
    
    bdat.to_csv(op.join(datapath, 'combined', 'EffortDifficulty_s%02d_combined_py.csv'%isub), index=False) #write this to csv, dont write row names
        
#%%

#collate all into one dataframe

fileList = sorted([x for x in os.listdir(op.join(datapath, 'combined')) if 'csv' in x])

df = pd.concat([pd.read_csv(op.join(datapath, 'combined', x)) for x in fileList])
df.to_csv(op.join(datapath, 'EffortDifficulty_maindata_py.csv'), index=False)
