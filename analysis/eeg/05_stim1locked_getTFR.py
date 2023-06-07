#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:00:39 2023

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

sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
from funcs import getSubjectInfo, gesd, plot_AR


wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)


subs = np.array([10, 11])
for i in subs:
    for run in [2]: #[1, 2]
        for baselined in [False]:
            print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
            sub   = dict(loc = 'laptop', id = i)
            param = getSubjectInfo(sub)
            
            if run == 1:
                #run on all trials epoched data
                in_fname  = param['stim1locked']
                out_fname = in_fname.replace('-epo.fif', '-tfr.h5')
            elif run == 2:
                #run on cleaned epoched data
                in_fname  = param['stim1locked'].replace('stim1locked', 'stim1locked_cleaned')
                out_fname = in_fname.replace('-epo.fif', '-tfr.h5')
                
            if baselined:
                out_fname = out_fname.replace('-tfr', '_baselined-tfr')
        
            epoched = mne.epochs.read_epochs(fname = in_fname, preload=True) #read raw data
            if baselined:
                epoched = epoched.apply_baseline((-0.2,0))
            epoched.set_eeg_reference(ref_channels='average') #set average electrode reference
            epoched.resample(100) #downsample to 100Hz so don't overwork the workstation
            #data are re-referenced to average mastoid reference during preprocessing
        
            #check to see if there are any trials where the previous trial confdiff is nan (i.e. no previous trial bcos start of session or block)
        
            # set up params for TF decomp
            freqs = np.arange(1, 41, 1)  # frequencies from 2-35Hz
            n_cycles = freqs *.3  # 300ms timewindow for estimation
        
            print('\nrunning TF decomposition\n')
            # Run TF decomposition overall epochs
            # tfr = mne.time_frequency.tfr_morlet(stim1locked, freqs=freqs, n_cycles=n_cycles,
            #                      use_fft=True, return_itc=False, average=False)
            tfr = mne.time_frequency.tfr_multitaper(epoched, freqs=freqs, n_cycles=n_cycles,
                                                    use_fft=True, return_itc=False, average=False)
            tfr.metadata.to_csv(param['behaviour'].replace('combined.csv', 'combined_stim1lockedtfr.csv'), index=False)
            print('\nSaving TFR data')
            tfr.save(fname = out_fname, overwrite = True)
            
            print('saving only alpha frequency data')
            tfr.crop(fmin = 8, fmax = 12)
            tfr.save(fname = out_fname.replace('-tfr.h5', '_Alpha-tfr.h5'))
        
            del(epoched)
            del(tfr)