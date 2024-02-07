#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:17:31 2024

@author: sammichekroud
"""

import numpy as np
import scipy as sp
import mne
import copy
from copy import deepcopy

def smooth(signal, twin = 50, method = 'boxcar'):
    '''

    function to smooth a signal. defaults to a 50ms boxcar smoothing (so quite small), just smooths out some of the tremor in the trace signals to clean it a bit
    can change the following parameters:

    twin    -- number of samples (if 1KHz sampling rate, then ms) for the window
    method  -- type of smoothing (defaults to a boxcar smoothing) - defaults to a boxcar
    '''
    if method == 'boxcar':
        #set up the boxcar
        filt = sp.signal.windows.boxcar(twin)

    #smooth the signal
    if method == 'boxcar':
        smoothed_signal = np.convolve(filt/filt.sum(), signal, mode = 'same')

    return smoothed_signal

def find_blinks(raw, buffer = 0.150, samplerate = 1000, add_nanchannel = True, add_transformed = False):
    '''

    Parameters
    ----------
    raw : RawEyelink
        A RawEyelink object - the output from mne.io.read_raweyelink. This should not contain blink annotations
    buffer : float, optional
        the buffer around blinks that you want to also mark to be interpolated. The default is 0.150, in seconds.
    samplerate : numeric, optional
        the sampling frequency of your data. This determines how many samples around blinks are set to nan. The default is 1000.
    add_nanchannel : TYPE, optional
        Whether or not to add in a channel to your data that contains the preprocessed data (noisy segments set to nan). The default is True.

    Returns
    -------
    data : RawEyelink
        Returns RawEyelink object.

    '''
    
    
    data = raw.copy()
    blinkspd        = 2.5                                   #speed above which data is remove around nan periods -- threshold
    maxvelthresh    = 30
    maxpupilsize    = 20000
    cleanms         = buffer * samplerate                   #ms padding around the blink edges for removal
    
    pupil_channel = [x for x in data.ch_names if 'pupil' in x][0]
    nonpup_chans = [x for x in data.ch_names if 'pupil' not in x]
    pupil = np.squeeze(data.copy().drop_channels(nonpup_chans).get_data())
    signal = pupil.copy()

    vel       = np.diff(pupil) #derivative of pupil diameter (velocity)
    speed     = np.abs(vel)    #absolute velocity
    smoothv   = smooth(vel, twin = 8, method = 'boxcar') #smooth with a 8ms boxcar to remove tremor in signal
    smoothspd = smooth(speed, twin = 8, method = 'boxcar') #smooth to remove some tremor
    
    #pupil diam shouldn't really reach zero, this is missing data, so set a check))
    zerosamples = np.zeros_like(pupil, dtype=bool)
    zerosamples[pupil==0] = True
    
    badsamples = np.zeros_like(pupil, dtype=bool)
    badsamples[1:] = np.logical_or(speed >= maxvelthresh, pupil[1:] > maxpupilsize)
    
    #if you smooth with a boxcar of width your buffer, it spreads the 1s (marker of bad sample) to the buffer period around (1/buffer width)
    #if you then just check for greater than zero, this gets all samples within the contaminated window 
    badsamples = np.greater(smooth(badsamples.astype(float), twin = int(cleanms), method='boxcar'), 0).astype(bool)
    badsamps = (badsamples | zerosamples) #get all bad samples before we find 'blink' periods that need interpolating
    
    signal[badsamps==1] = np.nan #set bad samples or zero samples to nan, this does a lot of preproc
    # plt.figure(); plt.plot(pupil, label='raw', color='b'); plt.plot(signal, label='preproc', color='r'); plt.legend(loc='lower left')
    
    changebads = np.zeros_like(pupil, dtype=int)
    changebads[1:] = np.diff(badsamps.astype(int)) #+1 = from not missing -> missing; -1 = missing -> not missing
    
    #starts are always off by one sample - when changebads == 1, the data is now MISSING. we need the sample before for interpolation
    starts = np.squeeze(np.where(changebads==1)) -1
    ends = np.squeeze(np.where(changebads==-1))
    
    if starts.size != ends.size:
        print(f"There is a problem with your data and the start/end of blinks dont match.\n- There are {starts.size} blink starts and {ends.size} blink ends")
        if starts.size == ends.size - 1:
            print('The recording starts on a blink; fixing')
            starts = np.insert(starts, 0, 0, 0)
        if starts.size == ends.size + 1:
            print('The recording ends on a blink; fixing')
            ends = np.append(ends, len(pupil))
    
    durations = np.divide(np.subtract(ends, starts), samplerate) #get duration of each saccade in samples
    
    blinkevents = np.array([starts, np.zeros(starts.size), np.ones(starts.size)]).T
    tmpannots = mne.annotations_from_events(blinkevents, sfreq = samplerate,
                                            orig_time = data.annotations[0]['orig_time'])
    tmpannots.duration = durations #set saccade durations
    tmpannots.description = np.tile(np.array(['BAD_blink']), starts.size)
    data.set_annotations(raw.annotations + tmpannots) #add the new blink  annotations into the raw data structure
    
    if add_nanchannel:
        tmp = mne.io.RawArray(signal.reshape(1, -1), info = data.copy().drop_channels(nonpup_chans).info)
        tmp.rename_channels({tmp.info.ch_names[0] : 'pupil_nan'}) #rename this channel (with data set to nan) before adding into raw object
        data.add_channels([tmp])
        
    if add_transformed:
        transformed_signal = np.zeros_like(signal)
        transformed_signal = np.divide(np.subtract(signal.copy(), np.nanmean(signal.copy())), np.nanmean(signal.copy())) * 100
        tmp = mne.io.RawArray(transformed_signal.reshape(1, -1), info = raw.copy().drop_channels(nonpup_chans).info)
        tmp.rename_channels({tmp.info.ch_names[0]:'pupil_transformed'})
        data.add_channels([tmp])
    
    #find location of blink annotations
    blinkannots = [x for x in range(len(data.annotations)) if 'BAD_blink' in data.annotations[x]['description']] #indices of blink annotations
    chanlabels = [x for x in data.ch_names if 'DIN' not in x] # all the non-DIN channel labels
    for x in blinkannots: #loop over blink annotations
        data.annotations.ch_names[x] = tuple(chanlabels) #add all non-DIN channels to the channel names related to this blink artefact label
    
    return data

def transform_pupil(raw):
    data = raw.copy()
    nonpup_chans = [x for x in data.ch_names if 'pupil' not in x]
    tmpdata = data.copy().drop_channels(nonpup_chans) #keep object only with the pupil data
    tmpdat  = tmpdata.get_data()  #preserves order of channels from tmpdata.ch_names
    
    mean_tmpdat = np.nanmean(tmpdat, axis=1).reshape(tmpdat.shape[0], -1)
    
    transformed = tmpdat.copy()
    transformed = np.subtract(tmpdat, mean_tmpdat) #demean
    transformed = np.multiply(np.divide(transformed, mean_tmpdat.reshape(tmpdat.shape[0], -1)),100) #express as a percentage of mean pupil
    
    #create temporary object to allow us to add the channels into the main data structure
    tmp = mne.io.RawArray(transformed, info = tmpdata.copy().info)
    rename_dict = dict()
    for name in tmpdata.ch_names:
        rename_dict[name] = name+'_transformed'
    tmp.rename_channels(rename_dict)
    data.add_channels([tmp])
    
    chanlabels = tuple([x for x in data.ch_names if 'DIN' not in x])
    blinks = [x for x in range(len(data.annotations)) if 'blink' in data.annotations[x]['description']]
    for blink in blinks:
        data.annotations.ch_names[blink] = chanlabels
    
    return data
