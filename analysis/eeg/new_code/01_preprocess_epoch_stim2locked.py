#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:07:43 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import sklearn as skl
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
%matplotlib
mne.viz.set_browser_backend('qt')

loc = 'laptop'
if loc == 'laptop':
    sys.path.insert(0, '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty'
elif loc == 'workstation':
    sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
    wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)

from funcs import getSubjectInfo

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
# subs = np.array([                                                                                                    35, 36, 37, 38, 39])
runanyway = True

#need to get subjects 13 and 14 data (not on computer) and rerun for them
#%% new pipeline
#1 - rereference
#2 - bandpass filter 1-40
#3 - create eog channels
#4 - epoch
#5 - ICA on epoched data

#some parameters for epoching
events_s1       = [1, 10] #stim1 triggers
events_s2       = [2, 3, 11, 12] #stim2 triggers
#epoching to stim1 onset
tmin, tmax = -3, 4
baseline = None #don't baseline just yet
#get events from annotations
event_id = { '1':   1, '10': 10,        #stim1

             '2':   2,  '3':  3,       #stim2
             '11': 11, '12': 12, 
             
             '22': 22, '23': 23,  #response cue
             '31': 31, '32': 32,  
             
             '40': 40, #button pressleft
             '50': 50, #button press right
             
              '60':  60, #correct feedback
              '61':  61, #incorrect feedback
              '62':  62, #timed out feedback,
             '150': 150, #ITI trigger
             }

#skipped 29 and 30 for now, need to amend script
for i in subs:
    sub   = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    print(f'- - - - working on subject {i} - - - -')
    # if not op.exists(param['eeg_preproc']) and i not in [29, 30]:  
    if i not in [29, 30]:
        raw = mne.io.read_raw_curry(fname = param['raweeg'], preload = True)
        
        raw = mne.add_reference_channels(raw, ref_channels = 'LM', copy = False) #left mastoid was active reference, add it back in (empty channel)
        raw.set_eeg_reference(ref_channels = ['LM', 'RM']) # set average mastoid reference
        raw.filter(1, 40, n_jobs=3) #filter between 0.1 and 40 Hz
        #create bipolar eog channel from EOGL (lower eye electrode) and referencing it to FP1 (above left eye)
        raw = mne.set_bipolar_reference(raw, drop_refs = False, #keep these channels in the data so we keep FP1
                                        cathode = 'EOGL', anode = 'FP1', ch_name = 'VEOG') #does anode - cathode
        raw.set_channel_types(mapping = {
            'VEOG':'eog',
            'HEOG': 'eog',
            'LM':'misc', 
            'RM':'misc',
            'EOGL':'misc', 
            'Trigger':'misc'
            })
        raw.set_montage('easycap-M1', on_missing='raise', match_case = False)
        # raw.info['bads'] = deepcopy(param['badchans'])
        raw.info['bads'].extend(['AFz']) #AFz was the ground, we want to interpolate it
        raw.interpolate_bads()
    
        events, _ = mne.events_from_annotations(raw, event_id)
        if i == 12:
            #one trigger is labelled wrong 
            wrongid = np.where(events[:,0] ==1325388)[0] #it received a 10 when 11 was sent
            events[wrongid,2] = 11 #re-label this trigger
        
        epoched = mne.Epochs(raw, events, events_s2, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        bdata = pd.read_csv(param['behaviour'], index_col = None)
        if i == 37:
            bdata = bdata.query('blocknumber in [1,2,3]') #participant withdrew after 3 blocks of task
        
        epoched.metadata = bdata
        
        #run ica on the epoched data
        ica = mne.preprocessing.ICA(n_components = 0.95, #cap at 30 components to be useful
                                    method = 'infomax').fit(epoched, picks='eeg',
              reject_by_annotation = True)
        
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name = ['HEOG', 'VEOG'])
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
        ica.plot_scores(eog_scores, eog_inds)
                
        ica.plot_components(inst=epoched, contours = 0)
        print('\n\n- - - - - - - subject %d, %d components to remove: - - - - - - -\n\n'%(i, len(eog_inds)))
        
        comps2rem = input('components  remove: ') #need to separate component numbers by comma and space
        comps2rem = list(map(int, comps2rem.split(', ')))
        np.savetxt(fname = op.join(param['path'], 'removed_comps', 's%02d_removedcomps_stim2locked.txt'%i),
                   X = comps2rem, fmt = '%i') #record what components were removed
        ica.exclude.extend(comps2rem) #mark components for removal
        ica.apply(inst=epoched)
        
        epoched.save(fname = param['stim2locked'].replace('stim2locked', 'stim2locked_icaepoched'), fmt = 'double', overwrite = True)
        
        plt.close('all')
        del(raw)
        del(eog_epochs)
    
    elif i in [29, 30]:
        raw1 = mne.io.read_raw_curry(fname = param['raweeg'], preload = True)
        raw2 = mne.io.read_raw_curry(fname = param['raweeg'].replace('.dat', 'b.dat'), preload = True)
        
        
        raw1 = mne.add_reference_channels(raw1, ref_channels = 'LM', copy = False) #left mastoid was active reference, add it back in (empty channel)
        raw1.set_eeg_reference(ref_channels = ['LM', 'RM']) # set average mastoid reference
        raw1.filter(1, 40, n_jobs=3) #filter between 0.1 and 40 Hz
        #create bipolar eog channel from EOGL (lower eye electrode) and referencing it to FP1 (above left eye)
        raw1 = mne.set_bipolar_reference(raw1, drop_refs = False, #keep these channels in the data so we keep FP1
                                        cathode = 'EOGL', anode = 'FP1', ch_name = 'VEOG') #does anode - cathode
        raw1.set_channel_types(mapping = {
            'VEOG':'eog',
            'HEOG': 'eog',
            'LM':'misc', 
            'RM':'misc',
            'EOGL':'misc', 
            'Trigger':'misc'
            })
        raw1.set_montage('easycap-M1', on_missing='raise', match_case = False)
        # raw.info['bads'] = deepcopy(param['badchans'])
        raw1.info['bads'].extend(['AFz']) #AFz was the ground, we want to interpolate it
        raw1.interpolate_bads()
        
        raw2 = mne.add_reference_channels(raw2, ref_channels = 'LM', copy = False) #left mastoid was active reference, add it back in (empty channel)
        raw2.set_eeg_reference(ref_channels = ['LM', 'RM']) # set average mastoid reference
        raw2.filter(1, 40, n_jobs=3) #filter between 0.1 and 40 Hz
        #create bipolar eog channel from EOGL (lower eye electrode) and referencing it to FP1 (above left eye)
        raw2 = mne.set_bipolar_reference(raw2, drop_refs = False, #keep these channels in the data so we keep FP1
                                        cathode = 'EOGL', anode = 'FP1', ch_name = 'VEOG') #does anode - cathode
        raw2.set_channel_types(mapping = {
            'VEOG':'eog',
            'HEOG': 'eog',
            'LM':'misc', 
            'RM':'misc',
            'EOGL':'misc', 
            'Trigger':'misc'
            })
        raw2.set_montage('easycap-M1', on_missing='raise', match_case = False)
        # raw.info['bads'] = deepcopy(param['badchans'])
        raw2.info['bads'].extend(['AFz']) #AFz was the ground, we want to interpolate it
        raw2.interpolate_bads()
    
        
        events1, _ = mne.events_from_annotations(raw1, event_id)
        events2, _ = mne.events_from_annotations(raw2, event_id)
        
        epoched1 = mne.Epochs(raw1, events1, events_s2, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        epoched2 = mne.Epochs(raw2, events2, events_s2, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        
        bdata = pd.read_csv(param['behaviour'], index_col = None)
        
        if i == 29:
            datdir = op.join(wd,'data', 'datafiles', 's29')
            filelist = os.listdir(datdir)
            flist1 = [x for x in filelist if 's29b' not in x]
            flist2 = [x for x in filelist if 's29b'      in x]
            
            #first recording has 2 full blocks and a bit of a third block, need to trim this down to just the complete blocks
            bdata1 = pd.DataFrame()
            for file in flist1:
                tmpdf = pd.read_csv(op.join(datdir, file), index_col=False)
                bdata1 = pd.concat([bdata1, tmpdf])
            bdata1 = bdata[:len(epoched1.events)] #cut this down to size
            epoched1.metadata = bdata1
            epoched1 = epoched1['blocknumber in [1,2]'] #cut down to just the first two blocks. should have 128 trials now
            epoched1.metadata = bdata.copy().query('blocknumber in [1,2]') #assign this back in now as it contains some extra columns that are useful
            #second recording has 3 full blocks
            bdata2 = bdata.query('blocknumber in [3,4,5]') #this we can just take from the combined datafile as it's accurate
            epoched2.metadata = bdata2
            
            # epoched = mne.concatenate_epochs([epoched1, epoched2]) #combine back now
        if i ==30:
            bdata1 = bdata.copy().query('blocknumber in [1,2,3]')
            bdata2 = bdata.copy().query('blocknumber in [4,5]')
            
            epoched1.metadata = bdata1
            epoched2.metadata = bdata2
        
        
        #ica the two datasets separately as they were recorded separately
        #run ica on the epoched data
        ica1 = mne.preprocessing.ICA(n_components = 0.95, #cap at 30 components to be useful
                                    method = 'infomax').fit(epoched1, picks='eeg',
              reject_by_annotation = True)
        
        eog_epochs = mne.preprocessing.create_eog_epochs(raw1, ch_name = ['HEOG', 'VEOG'])
        eog_inds, eog_scores = ica1.find_bads_eog(eog_epochs)
        ica1.plot_scores(eog_scores, eog_inds)
                
        ica1.plot_components(inst=epoched1, contours = 0)
        print('\n\n- - - - - - - subject %d part 1, %d components to remove: - - - - - - -\n\n'%(i, len(eog_inds)))
        comps2rem = input('components  remove: ') #need to separate component numbers by comma and space
        comps2rem = list(map(int, comps2rem.split(', ')))
        np.savetxt(fname = op.join(param['path'], 'removed_comps', 's%02d_removedcomps_stim2locked.txt'%i),
                   X = comps2rem, fmt = '%i') #record what components were removed
        ica1.exclude.extend(comps2rem) #mark components for removal
        ica1.apply(inst=epoched1)
        
        
        ica2 = mne.preprocessing.ICA(n_components = 0.95,
                                     method = 'infomax').fit(epoched2, picks = 'eeg',
                                                             reject_by_annotation = True)
        eog_epochs = mne.preprocessing.create_eog_epochs(raw2, ch_name = ['HEOG', 'VEOG'])
        eog_inds, eog_scores = ica2.find_bads_eog(eog_epochs)
        ica2.plot_scores(eog_scores, eog_inds)
                
        ica2.plot_components(inst=epoched2, contours = 0)
        print('\n\n- - - - - - - subject %d part 2, %d components to remove: - - - - - - -\n\n'%(i, len(eog_inds)))
        
        comps2rem = input('components  remove: ') #need to separate component numbers by comma and space
        comps2rem = list(map(int, comps2rem.split(', ')))
        np.savetxt(fname = op.join(param['path'], 'removed_comps', 's%02db_removedcomps_stim2locked.txt'%i),
                   X = comps2rem, fmt = '%i') #record what components were removed
        ica2.exclude.extend(comps2rem) #mark components for removal
        ica2.apply(inst=epoched2)
        
        epoched = mne.concatenate_epochs([epoched1, epoched2])
        
        epoched.save(fname = param['stim2locked'].replace('stim2locked', 'stim2locked_icaepoched'), fmt = 'double', overwrite = True)
        
        plt.close('all')
        del(raw1)
        del(raw2)
        del(ica1)
        del(ica2)
        del(eog_epochs)

#%%
#to resample and re-save the already ica cleaned data...
for i in subs:
    sub   = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    print(f'- - - - working on subject {i} - - - -')
    
    epoched = mne.read_epochs(fname = param['stim2locked'].replace('stim2locked', 'stim2locked_icaepoched'), preload =True)
    epoched.resample(500) #downsample to 500hz to save space
    epoched.save(fname = param['stim2locked'].replace('stim2locked', 'stim2locked_icaepoched_resamp'), fmt = 'double', overwrite = True)