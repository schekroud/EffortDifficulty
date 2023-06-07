#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:24:54 2023

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
import seaborn as sns
np.set_printoptions(suppress=True)
import re
%matplotlib

# sys.path.insert(0, '/ohba/pi/knobre/schekroud/postdoc/student_projects/perceptdiff/analysis_scripts/tools')
# from funcs import getSubjectInfo
# import funcs
# import eyefuncs as eyes


# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)


def checkTriggers(events, bdata):
    #triggers in the task
    ititrigs            = [150]
    stim1trigs          = [1,10]
    stim2trigs          = [2, 3, 11, 12] #stim2 triggers
    respcuetrigs        = [22, 23, 31, 32] #response cue triggers
    buttonlefttrigs     = [40] #button press left triggers
    buttonrighttrigs    = [50] #button press right triggers
    buttontrigs         = [40, 50]
    fbtrigs             = [60, 61, 62]
    
    eventdf = pd.DataFrame(events, columns = ['onset', 'duration','trigger'])
    ntrls=len(bdata) #get number of trials performed
    
    itis  = eventdf.query('trigger in @ititrigs')
    stim1s = eventdf.query('trigger in @stim1trigs')
    stim2s = eventdf.query('trigger in @stim2trigs')
    respcs = eventdf.query('trigger in @respcuetrigs')
    buttons = eventdf.query('trigger in @buttontrigs')
    fbs = eventdf.query('trigger in @fbtrigs')
    
    #run some checks based on behavioural data, print some messages
    
    #check itis match number of triggers sent
    if len(itis) == ntrls:
        print('\n- no missing ITI triggers')
    else:
        print('\n- missing %02d ITI triggers'%(ntrls-len(itis)))
        
    if len(stim1s) == ntrls:
        print('\n- no missing stim1triggers')
        if np.equal(stim1s.trigger.to_numpy(), bdata.stim1trig.to_numpy()).sum() == ntrls:
            print('--- received stim1 triggers match triggers in behavioural data')
        else:
            print('--- received stim1 triggers *do not* match triggers in behavioural data')
    else:
        print('\n- missing %02d stim1 triggers'%ntrls - len(stim1s))
    
    if len(stim2s) == ntrls:
        print('\n- no missing stim2 triggers')
        if np.equal(stim2s.trigger.to_numpy(), bdata.stim2trig.to_numpy()).sum() == ntrls:
            print('--- received stim2 triggers match triggers in behavioural data')
        else:
            print('--- received stim2 triggers *do not* match triggers in behavioural data')
    else:
        print('\n- missing %02d stim2 triggers'%ntrls - len(stim2s))
    
    if len(respcs) == ntrls:
        print('\n- no missing respcue triggers')
        if np.equal(respcs.trigger.to_numpy(), bdata.respcuetrig.to_numpy()).sum() == ntrls:
            print('--- received respcue triggers match triggers in behavioural data')
        else:
            print('--- received respcue triggers *do not* match triggers in behavioural data')
    else:
        print('\n- missing %02d respcue triggers'%ntrls - len(respcs))
    
    if len(fbs) == ntrls:
        print('\n- no missing feedback triggers')
        if np.equal(fbs.trigger.to_numpy(), bdata.fbtrig.to_numpy()).sum() == ntrls:
            print('--- received feedback triggers match triggers in behavioural data')
        else:
            print('--- received feedback triggers *do not* match triggers in behavioural data')
    else:
        print('\n- missing %02d feedback triggers'%ntrls - len(fbs))

#%%
# ntrls = 320 #should be 320 trials for all things

subid = 8
print('running for subject %s\n'%str(subid))
timingtestfolder = op.join(wd, 'timing_tests')
datafolder = op.join(wd, 'data')
logfolder = op.join(datafolder, 'logs')
eegfolder = op.join(wd, 'eeg', 'timing_tests')

eegname = op.join(eegfolder, 's%02d_timingcheck.dat'%subid)

raw = mne.io.read_raw_curry(
        fname = eegname, preload = False, verbose = False)

bdatadir = op.join(wd, 'data', 'datafiles', 's%02d'%subid)
files = sorted(os.listdir(bdatadir))
# files = sorted([x for x in files if 'barStim' in x])
bdata = pd.DataFrame()
for file in files:
    tmpdf = pd.read_csv(op.join(bdatadir, file))
    bdata = pd.concat([bdata, tmpdf])
ntrls = bdata.shape[0]

event_id = { '1':  1, '10': 10, #stim1
             
             '2':  2,   '3':  3,       #stim2
             '11': 11, '12': 12, 
             #these triggers are randomly added into some subjects?
             
             '22':   22,  '23':  23,  #response cue
             '31':   31,  '32':  32,  
             
             '40':   40, #button pressleft
             '50':  50, #button press right
             
             '60': 60, #correct feedback
             '61': 61, #incorrect feedback
             '62': 62, #timed out feedback,
             # '254':254, #used in barStim test for the fake response
             # '255':255, #it changed it for some reason
             '150':150, #ITI trigger
             '200':200 #PLR trigger
             }

events,_ = mne.events_from_annotations(raw, event_id = event_id)
# events_eyes = deepcopy(events) #for the alternative method of fixing

checkTriggers(events, bdata)



#%%
#load in the timing log to see if this is also the case here (as same format code used to send triggers)

logname = op.join(logfolder, 'EffortDifficulty_S%02da_logfile.txt'%(subid))
    
d = open(logname, 'r')
raw_d = d.readlines()
d.close()

log = [x for x in raw_d if 'trial' in x]
log = [x.replace(' ', '') for x in log]

regex = "(\d+.\d+)\\t(EXP)\\t(trial)([\d]+)_(trig)(\d+)\\n"
p = re.compile(regex)


log = [list(p.match(x).groups()) for x in log]
log = np.array(log)
log

logdf = pd.DataFrame(log, columns = ['onset', 'exp', 'trial', 'trlid', 'trig', 'triggerid'])
logdf.onset = logdf.onset.astype(float)
logdf.triggerid = logdf.triggerid.astype(float)


ititrig         = ['trig150', 'iti_start']
stim1trigs      = ['trig1', 'trig10']
stim2trigs      = ['trig2','trig3','trig11','trig12']
respcuetrigs    = ['trig22','trig23', 'trig31','trig32']
# buttontrig      = ['trig254']
buttontrig      = ['trig40', 'trig50', 'trig254', 'trig255']
fbtrigs         = ['trig60','trig61', 'trig62']


ititriggers = logdf.query('triggerid in @ititrigs')
s1triggers  = logdf.query('triggerid in @stim1trigs')
s2triggers  = logdf.query('triggerid in @stim2trigs')
rcuetriggers  = logdf.query('triggerid in @respcuetrigs')
buttriggers  = logdf.query('triggerid in @buttontrigs')
feedbtriggers  = logdf.query('triggerid in @fbtrigs')

