#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 22:53:51 2023

@author: sammi
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

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)


subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
for i in subs:
    sub   = dict(loc = 'workstation', id = i)
    param = getSubjectInfo(sub)
    if not op.exists(param['eeg_preproc']):      
        raw = mne.io.read_raw_curry(fname = param['raweeg'], preload = True)
        
        raw = mne.add_reference_channels(raw, ref_channels = 'M2', copy = False) #left mastoid was active reference, add it back in (empty channel)
        raw.rename_channels({'RM':'M1'})
        
        #create bipolar eog channel from EOGL (lower eye electrode) and referencing it to FP1 (above left eye)
        raw = mne.set_bipolar_reference(raw, drop_refs = False, #keep these channels in the data so we keep FP1
                                        cathode = 'EOGL', anode = 'FP1', ch_name = 'VEOG') #does anode - cathode
        raw.set_channel_types(mapping = {
            'VEOG':'eog',
            'HEOG': 'eog',
            'M1':'misc', 
            'M2':'misc',
            'EOGL':'misc', 
            'Trigger':'misc'
            })
        
        raw.set_montage('easycap-M1', on_missing='raise', match_case = False)
        
        reftype = 'average'
        if reftype == 'mastoid':
            raw.filter(1, 40, n_jobs=3) #filter between 0.1 and 40 Hz
            raw.set_eeg_reference(ref_channels = ['M1', 'M2']) # set average mastoid reference
            #this filters bad channels too, which is fine
            raw.info['bads'] = deepcopy(param['badchans'])
            raw.info['bads'].extend(['AFz']) #AFz was the ground, we want to interpolate it
        elif reftype == 'average':
            #set bad channels here so they are ignored in the average referencing
            raw.filter(1, 40, picks = 'all', n_jobs=3) #make sure that even 'bad' channels are filtered too (no real reason not to)
            #first re-ref to average mastoid to reduce asymmetry in the reference
            raw.set_eeg_reference(ref_channels = ['M1', 'M2'])
            
            raw.info['bads'] = deepcopy(param['badchans'])
            raw.info['bads'].extend(['AFz']) #AFz was the ground, we want to interpolate it
            raw.set_eeg_reference(ref_channels = 'average') #then re-reference to the average
        
        raw.interpolate_bads() #interpolate bad channels
        
        
        #if you need to remove break periods from the data to make sure that the ica doesnt get messed up
        # (extended break periods have high noise and influence the ICA)
        #then you can browse the raw data and mark it to be ignored
        # raw.plot(duration = 60, n_channels = 63) #browse in units of 1min
        # raw.interpolate_bads() #if you need to interpolate any channels you see are bad during the data review
        
        # this will mark periods between the last event of a block and the first event of the next block as a break
        # to exclude from preproc so it doesn't cause problems by introducing lots of noise
        break_annotations = mne.preprocessing.annotate_break(raw,
                                               t_start_after_previous= 4,
                                               t_stop_before_next = 4) #give at least 4s after last event and 4s before next
        raw.set_annotations(raw.annotations + break_annotations) #combine annotations here
        
        #run ica
        ica = mne.preprocessing.ICA(n_components = 30, #cap at 30 components to be useful
                                    method = 'infomax').fit(raw,
              reject_by_annotation = True) #this will exclude break periods if marked as bad using the code above
              # reject = dict(eeg = 400e-6))
        # ideally this just excludes periods between blocks where ppts took breaks as they move a lot then
        
        
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name = ['HEOG', 'VEOG'])
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
        ica.plot_scores(eog_scores, eog_inds)
        
        ica.plot_components(inst=raw, contours = 0)
        print('\n\n- - - - - - - subject %d, %d components to remove: - - - - - - -\n\n'%(i, len(eog_inds)))
        
        comps = ica.get_sources(inst=raw)
        c = comps.get_data()
        rsac = np.empty(shape = (c.shape[0]))
        rblk = np.empty(shape = (c.shape[0]))
        heog = raw.get_data(picks = 'HEOG').squeeze()
        veog = raw.get_data(picks = 'VEOG').squeeze()
        for comp in range(len(c)):
            rsac[comp] = sp.stats.pearsonr(c[comp,:], heog)[0]
            rblk[comp] = sp.stats.pearsonr(c[comp,:], veog)[0]
        
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.bar(np.arange(len(c)), rsac)
        ax.set_title('corr with heog')
        
        ax2 = fig.add_subplot(2,1,2)
        ax2.bar(np.arange(len(c)), rblk)
        ax2.set_title('corr with veog')
        fig.tight_layout()
       
        plt.pause(3)
        
        comps2rem = input('components  remove: ') #need to separate component numbers by comma and space
        comps2rem = list(map(int, comps2rem.split(', ')))
        np.savetxt(fname = op.join(param['path'], 'removed_comps', 's%02d_removedcomps.txt'%i),
                   X = comps2rem, fmt = '%i') #record what components were removed
        ica.exclude.extend(comps2rem) #mark components for removal
        ica.apply(inst=raw)
        
        raw.save(fname = param['eeg_preproc'], fmt = 'double', overwrite = True)
        plt.close('all')
        del(comps)
        del(c)
        del(raw)
        del(eog_epochs)
    