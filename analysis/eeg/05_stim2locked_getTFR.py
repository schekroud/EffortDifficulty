# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:13:14 2023

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

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/analysis/tools')
sys.path.insert(0, 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools')

from funcs import getSubjectInfo, gesd, plot_AR

# wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
# wd = 'C:/Users/sammi/Desktop/Experiments/postdoc/student_projects/EffortDifficulty/'
wd = 'C:/Users/sammirc/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd

os.chdir(wd)

subs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

for i in subs:
    for run in [1,2]: #[1, 2]
        for baselined in [False]:
            print('\n- - - - working on subject %s - - - - -\n'%(str(i)))
            sub   = dict(loc = 'workstation', id = i)
            param = getSubjectInfo(sub)
            
            if run == 1:
                #run on all trials epoched data
                in_fname  = param['stim2locked']
                out_fname = in_fname.replace('-epo.fif', '-tfr.h5')
            elif run == 2:
                #run on cleaned epoched data
                in_fname  = param['stim2locked'].replace('stim2locked', 'stim2locked_cleaned')
                out_fname = in_fname.replace('-epo.fif', '-tfr.h5')
                
            if baselined:
                out_fname = out_fname.replace('-tfr', '_baselined-tfr')
        
            epoched = mne.epochs.read_epochs(fname = in_fname, preload=True) #read raw data
            if baselined:
                epoched = epoched.apply_baseline((0,0.2))
                
            # epoched.set_eeg_reference(ref_channels='average') #set average electrode reference
            epoched.resample(100) #downsample to 100Hz so don't overwork the workstation
             
            # set up params for TF decomp
            freqs = np.arange(1, 41, 1)  # frequencies from 1-40Hz
            n_cycles = freqs *.3  # 300ms timewindow for estimation
        
            print('\nrunning TF decomposition\n')
            # Run TF decomposition overall epochs
            # tfr = mne.time_frequency.tfr_morlet(stim1locked, freqs=freqs, n_cycles=n_cycles,
            #                      use_fft=True, return_itc=False, average=False)
            tfr = mne.time_frequency.tfr_multitaper(epoched, freqs=freqs, n_cycles=n_cycles,
                                                    use_fft=True, return_itc=False, average=False, n_jobs = 2)
            tfr.metadata.to_csv(param['behaviour'].replace('combined.csv', 'combined_stim2lockedtfr.csv'), index=False)
            print('\nSaving TFR data')
            tfr.save(fname = out_fname, overwrite = True)
            
            print('saving only alpha frequency data')
            tfr.crop(fmin = 8, fmax = 12)
            tfr.save(fname = out_fname.replace('-tfr.h5', '_Alpha-tfr.h5'), overwrite = True)
        
            del(epoched)
            del(tfr)