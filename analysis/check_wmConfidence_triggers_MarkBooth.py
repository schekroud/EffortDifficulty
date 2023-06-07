#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:49:01 2023

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

# sys.path.insert(0, '/ohba/pi/knobre/schekroud/postdoc/student_projects/perceptdiff/analysis_scripts/tools')
# from funcs import getSubjectInfo
import funcs
# import eyefuncs as eyes


# wd = '/ohba/pi/knobre/schekroud/postdoc/student_projects/perceptdiff' #workstation wd
wd = '/Users/sammi/Desktop/postdoc/wmconfidence' #laptop dir
os.chdir(wd)
# eyedir = op.join(wd, 'data', 'eyes')

fname = op.join(wd, 'data', 'eeg_test', 'wmconfidence_blankRecording1.dat')

raw = mne.io.read_raw_curry(
        fname =fname, preload = False, verbose = False)

bdatadir = op.join(wd, 'data', 'datafiles', 's60')
files = sorted(os.listdir(bdatadir))
df = pd.DataFrame()
for file in files:
    tmpdf = pd.read_csv(op.join(bdatadir, file))
    df = pd.concat([df, tmpdf])
ntrls = df.shape[0]

arraytrigs          = [1, 2]
cuetrigs          = [11, 12, 13, 14] # cue triggers
prbontrigs        = [21, 22, 23, 24] #probe cue triggers
moveontrigs     = [31, 32, 33, 34] #move onset triggers
moveofftrigs    = [41, 42, 43, 44] #move offset triggers
confontrigs     = np.add(moveofftrigs, 10)
confmovontrigs  = np.add(confontrigs, 10)
confmovofftrigs = np.add(confmovontrigs, 10)
fbtrigs         = np.add(confmovofftrigs, 5)

event_id = dict()
for i in arraytrigs:
    event_id[str(i)] = i
for i in cuetrigs:
    event_id[str(i)] = i
for i in prbontrigs:
    event_id[str(i)] = i
for i in moveontrigs:
    event_id[str(i)] = i
for i in moveofftrigs:
    event_id[str(i)] = i
for i in confontrigs:
    event_id[str(i)] = i
for i in confmovontrigs:
    event_id[str(i)] = i
for i in confmovofftrigs:
    event_id[str(i)] = i
for i in fbtrigs:
    event_id[str(i)] = i

event_id['255'] = 255
event_id['254'] = 254


events,_ = mne.events_from_annotations(raw, event_id = event_id)
# events_eyes = deepcopy(events) #for the alternative method of fixing

trigarray = np.array([raw.annotations.description.copy().astype(int), raw.annotations.onset.copy()]).T #get trigger id and onset

triggers = pd.DataFrame(trigarray, columns= [ 'trigger','onset']) #make into dataframe
triggers.trigger = triggers.trigger.astype(int)



#some checks to see what triggers are missing
arraytriggers = triggers.query('trigger in @arraytrigs').reset_index(drop=True)
print('\n\nTRIGGERS FOR %s'%param['subid'])
print('first trigger in the data is: %s'%str(int(trigarray[0,0])))

nmissing_tstart  = ntrls - len(triggers.query('trigger == 150'))
nmissing_stim1   = ntrls - len(triggers.query('trigger in @arraytrigs'))
nmissing_stim2   = ntrls - len(triggers.query('trigger in @cuetrigs'))
nmissing_respcue = ntrls - len(triggers.query('trigger in @prbontrigs'))
nmissing_fb      = ntrls - len(triggers.query('trigger in @fbtrigs'))
nmissing_total = np.sum([nmissing_tstart, nmissing_stim1, nmissing_stim2, nmissing_respcue, nmissing_fb])


#get trigger numbers to see what is missing
if triggers.query('trigger == 150').shape[0] != ntrls:
    print('missing %s triggers for trial start'%(str(ntrls-len(triggers.query('trigger == 150')))))

if triggers.query('trigger in @stim1trigs').shape[0] != ntrls:
    print('missing %s triggers for stim1'%(str(ntrls-len(triggers.query('trigger in @stim1trigs')))))
else:
    print('no missing triggers for stim1')
    s1trigs = triggers.query('trigger in @stim1trigs').trigger.to_numpy()
    s1trigs_bdata = bdata.stim1trig.to_numpy()
    check_s1trigs = np.sum(np.equal(s1trigs, s1trigs_bdata))
    if check_s1trigs == ntrls: #no missing stim1 triggers and the received triggers are the same as the ones that were sent
        print('>\tstim1 triggers received match triggers stored in behavioural data')
    elif check_s1trigs != ntrls: #no missing stim1 triggers but the received triggers aren't the same as was sent
        print('>\terror: stim1 triggers received do NOT match the triggers stored in behavioural data')

if triggers.query('trigger in @stim2trigs').shape[0] != ntrls:
    print('missing %s triggers for stim2'%(str(ntrls-len(triggers.query('trigger in @stim2trigs')))))
    
if triggers.query('trigger in @respcuetrigs').shape[0] != ntrls:
    print('missing %s triggers for response cue'%(str(ntrls-len(triggers.query('trigger in @respcuetrigs')))))

if triggers.query('trigger in @fbtrigs').shape[0] != ntrls:
    print('missing %s triggers for feedback'%(str(ntrls-len(triggers.query('trigger in @fbtrigs')))))
    fbtriggers = triggers.query('trigger in @fbtrigs').trigger.to_numpy()
    fbtriggers_bdata = bdata.fbtrig.to_numpy()
    check_fbtrigs = np.sum(np.equal(fbtriggers, fbtriggers_bdata))
    if check_fbtrigs == ntrls:
        print('>\tfeedback triggers received match the triggers stored in behavioural data')
    elif check_fbtrigs != ntrls: #no missing fb triggers but the received triggers aren't the same as what was sent
        print('\terror: feedback triggers received do NOT match the triggers stored in behavioural data')
print('%s missing triggers in total'%str(nmissing_total))