#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:03:02 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pickle

class rawEyes:
    def __init__(self, nblocks, srate):
        self.nblocks = nblocks
        self.data    = list()
        self.srate   = srate
        self.fsamp   = None
        self.binocular = None
    
    def nan_missingdata(self):
        for iblock in range(self.nblocks): #loop over blocks
            tmpdata = self.data[iblock]
            missinds = np.where(tmpdata.pupil == 0) #missing data is assigned to 0 for pupil trace
            tmpdata.pupil[missinds] = np.nan #nan everything where the pupil is zero
            tmpdata.xpos[missinds]  = np.nan
            tmpdata.ypos[missinds]  = np.nan
            self.data[iblock] = tmpdata
    
    def find_blinks(self, buffer = 0.150, add_nanchannel = True):
        #set up some parameters for the algorithm
        blinkspd        = 2.5                     #speed above which data is remove around nan periods -- threshold
        maxvelthresh    = 30
        maxpupilsize    = 20000
        cleanms         = buffer * self.srate     #ms padding around the blink edges for removal
        
        for iblock in range(self.nblocks): #loop over blocks
            tmpdata = self.data[iblock]
            pupil  = tmpdata.pupil
            signal = pupil.copy()
            vel    = np.diff(pupil) #derivative of pupil diameter
            speed  = np.abs(vel)    #absolute velocity
            smoothv   = smooth(vel, twin = 8, method = 'boxcar') #smooth with a 8ms boxcar to remove tremor in signal
            smoothspd = smooth(speed, twin = 8, method = 'boxcar') #smooth to remove some tremor
            #not sure if it quantitatively changes anything if you use a gaussian instead. the gauss filter makes it smoother though
            
            #pupil size only ever reaches zero if missing data. so we'll log this as missing data anyways
            zerosamples = np.zeros_like(pupil, dtype=bool)
            zerosamples[pupil==0] = True
            
            #create an array logging bad samples in the trace
            badsamples = np.zeros_like(pupil, dtype=bool)
            badsamples[1:] = np.logical_or(speed >= maxvelthresh, pupil[1:] > maxpupilsize)
            
            #a quick way of marking data for removal is to smooth badsamples with a boxcar of the same width as your buffer.
            #it spreads the 1s in badsamples to the buffer period around (each value becomes 1/buffer width)
            #can then just check if badsamples > 0 and it gets all samples in the contaminated window
            badsamples = np.greater(smooth(badsamples.astype(float), twin = int(cleanms), method = 'boxcar'), 0).astype(bool)
            badsamps = (badsamples | zerosamples) #get whether its marked as a bad sample, OR marked as a previously zero sample ('blinks' to be interpolated)
            signal[badsamps==1] = np.nan #set these bad samples to nan
            
            #we want to  create 'blink' structures, so we need info here
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

            durations = np.divide(np.subtract(ends, starts), self.srate) #get duration of each saccade in seconds
            
            blinkarray = np.array([starts, ends, durations]).T
            
            tmpdata.blinks = Blinks(blinkarray)
            
            if add_nanchannel:
                tmpdata.pupil_nan = signal
            
            self.data[iblock] = tmpdata #update the data object in place
    
    def smooth_pupil(self, sigma = 50):
        '''
        smooth the clean pupil trace with a gaussian with standard deviation sigma
        '''
        
        for iblock in range(self.nblocks):
            self.data[iblock].pupil_clean = sp.ndimage.gaussian_filter1d(self.data[iblock].pupil_clean, sigma = sigma)
            
    def save(self, fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle)
            
    def cubicfit(self):
        #define cubic function to be fit to the data
        def cubfit(x, a, b, c, d):
            return a*np.power(x,3) + b*np.power(x, 2) + c*np.power(x,1) + d
        for iblock in range(self.nblocks):
            
            tmpdata = self.data[iblock]
            fitparams = sp.optimize.curve_fit(cubfit, tmpdata.time, tmpdata.pupil_clean)[0]
            modelled  = fitparams[0]*np.power(tmpdata.time, 3) + fitparams[1]*np.power(tmpdata.time, 2) + fitparams[2]*np.power(tmpdata.time, 1) + fitparams[3]
            diff = tmpdata.pupil_clean - modelled #subtract this cubic fit
            #assign modelled data and the corrected data back into the data structure
            
            self.data[iblock].modelled        = modelled
            self.data[iblock].pupil_corrected = diff 
    
    def transform_channel(self, channel, method = 'percent'):
        for iblock in range(self.nblocks): #loop over blocks in the data
            tmpdata = self.data[iblock].__dict__[channel].copy()
            transformed = np.zeros_like(tmpdata)
            if method == 'zscore':
                transformed = sp.stats.zscore(tmpdata)
            elif method == 'percent':
                mean = tmpdata.mean()
                transformed = np.subtract(tmpdata, mean)
                transformed = np.multiply(np.divide(transformed, mean), 100)
            self.data[iblock].pupil_transformed = transformed #save the transformed data back into the data object
            
        
            
        

class epochedEyes():
    def __init__(self, data, srate, events, times):
        self.data     = data
        self.srate    = srate
        self.event_id = events
        self.times    = times
        self.tmin     = times.min()
        self.metadata = None
        
    
    def apply_baseline(self, baseline):
        bmin, bmax = baseline[0], baseline[1]
        #get baseline value per trial
        blines = np.logical_and(np.greater_equal(self.times, bmin), np.less_equal(self.times, bmax))
        blinedata = self.data[:,blines].mean(axis=1) #average across time in this window
        for itrl in range(self.data.shape[0]):
            self.data[itrl] -= blinedata[itrl]
        
        return self
            
        

def epoch(data, tmin, tmax, triggers, channel = 'pupil_clean'):
    nblocks = data.nblocks
    srate = data.srate
    allepochs = []
    alltrigs  = []
    for iblock in range(nblocks):
        tmpdata = data.data[iblock]
        findtrigs = np.isin(tmpdata.triggers.event_id, triggers) #check if triggers are present
        epoched_events = tmpdata.triggers.event_id[findtrigs] #store the triggers that are found, in order
        trigttimes = tmpdata.triggers.timestamp[findtrigs]     #get trackertime for the trigger onset
        trigtimes = np.squeeze(np.where(np.isin(tmpdata.trackertime, trigttimes))) #get indices of the trigger onsets
        tmins = np.add(trigtimes, tmin*srate).astype(int) #enforce integer so you can use it as an index
        tmaxs = np.add(trigtimes, tmax*srate).astype(int) #enforce integer so you can use it as an index
        iepochs = np.zeros(shape = [trigtimes.size, np.arange(tmin, tmax, 1/srate).size])
        for itrig in range(tmins.size):
            iepochs[itrig] = tmpdata.__dict__[channel][tmins[itrig]:tmaxs[itrig]]
        allepochs.append(iepochs)
        alltrigs.append(epoched_events)
    stacked  = np.vstack(allepochs)
    alltrigs = np.hstack(alltrigs)
    epochtimes = np.arange(tmin, tmax, 1/srate)
    #round this to match the sampling rate
    epochtimes = np.round(epochtimes, 3) #round to the nearest milisecond as we dont record faster than 1khz
    
    #create new object
    epoched = epochedEyes(data = stacked, srate = srate, events = alltrigs, times = epochtimes)
    
    return epoched

class EyeTriggers():
    def __init__(self):
        self.timestamp = None
        self.event_id  = None

class EyeHolder:
    def __init__(self):
        # self.fsamp = None
        self.info      = dict()
        self.triggers  = EyeTriggers()
        self.binocular = None
        self.eyes_recorded = []

class Blinks:
    def __init__(self, blinkarray):
        self.nblinks = blinkarray.shape[0]
        self.blinkstart = blinkarray[:,0]
        self.blinkend   = blinkarray[:,1]
        self.blinkdur   = blinkarray[:,2]

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


def parse_eyes(fname, srate = 1000):#, binocular = False):
    # if binocular:
    #     eyedata = _parse_binocular(fname, srate)
    # elif not binocular:
    #     eyedata = _parse_monocular(fname, srate)
    eyedata = _parse_eyes(fname, srate)
    
    return eyedata

def _parse_eyes(fname, srate):
    ncols_search = [6, 9] #if monocular search for 6 columns, if binocular search for 9 columns
    d = open(fname, 'r')
    raw_d = d.readlines()
    d.close()
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    fstart = starts[0]
    raw_d = raw_d[fstart:] #remove calibration info at the beginning of the file opening
    
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    ends   = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'END']
    
    eyedata = rawEyes(nblocks=len(starts), srate = srate)
    eyedata.binocular=True #log that this *is* a binocular recording
    
    # by handling binocular/monocular separately for each block of the data
    # you can handle situations where you change binocular/monocular between task blocks
    
    nsegments = len(starts)
    for iseg in range(nsegments):
        print(f'parsing block {iseg+1}/{nsegments}')
        istart, iend = starts[iseg], ends[iseg]
        rdata = raw_d[istart:iend+1]
        startmsg = rdata[0].split() #parse the start message as this tells you how many eyes are recorded
        if 'LEFT' in startmsg and 'RIGHT' in startmsg:
            binoc = True
        else:
            binoc = False
        data = rdata[7:] #cut out some of the nonsense before recording starts
        idata = [x for x in data if len(x.split()) == ncols_search[int(binoc)]]
        msgs  = [x for x in data if len(x.split()) != ncols_search[int(binoc)]]
        idata = np.asarray([x.split() for x in idata])[:, :-1] #drop the last column as it is nonsense (it's just '.....')
        if iseg == 0:
            fsamp = int(idata[0][0]) #get starting sample number
            eyedata.fsamp = fsamp
        #missing data is coded as '.' in the asc file - we need to make this usable but easily identifiable
        idata = np.where(idata=='.', 'NaN', idata) #set this missing data to a string nan that numpy can handle easily
        idata = idata.copy().astype(float) #convert into numbers now
        
        segdata             = EyeHolder()
        segdata.binocular   = binoc
        if 'LEFT' in startmsg:
            segdata.eyes_recorded.append('left')
        if 'RIGHT' in startmsg:
            segdata.eyes_recorded.append('right')
        
        segdata.trackertime = idata[:,0]
        segdata.time        = np.subtract(segdata.trackertime, segdata.trackertime[0]) #time relative to the first sample
        
        if segdata.binocular: #if binocular, add both eyes
            colsadd = ['xpos_l', 'ypos_l', 'pupil_l', 'xpos_r', 'ypos_r', 'pupil_r']
            for icol in range(len(colsadd)):
                setattr(segdata, colsadd[icol], idata[:,icol+1])
        if not segdata.binocular and 'left' in segdata.eyes_recorded: #if only left recorded, add as left eye specifically
            colsadd = ['xpos_l', 'ypos_l', 'pupil_l']
            for icol in range(len(colsadd)):
                setattr(segdata, colsadd[icol], idata[:, icol+1])
        if not segdata.binocular and 'right' in segdata.eyes_recorded: #if only right recorded, add as right eye specifically
            colsadd = ['xpos_r', 'ypos_r', 'pupil_r']
            for icol in range(len(colsadd)):
                setattr(segdata, colsadd[icol], idata[:, icol+1])
        
        #some parsing of triggers etc
        msgs = [x.split() for x in msgs]
        triggers = np.array([x for x in msgs if x[0] == 'MSG'])
        segdata.triggers.timestamp = triggers[:,1].astype(int)
        segdata.triggers.event_id  = triggers[:,2]
        
        eyedata.data.append(segdata)
    return eyedata

def _parse_monocular(fname, srate):
    #by default it's just going to look for stop/starts in the data file
    d = open(fname, 'r')
    raw_d = d.readlines()
    d.close()
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    fstart = starts[0]
    raw_d = raw_d[fstart:] #remove calibration info at the beginning of the file opening
    
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    ends   = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'END']
    
    eyedata = rawEyes(nblocks=len(starts), srate = srate)
    eyedata.binocular = False #log that this is *not* a binocular recording
    nsegments = len(starts)
    for iseg in range(nsegments):
        print(f'parsing block {iseg+1}/{nsegments}')
        istart, iend = starts[iseg], ends[iseg]
        data = raw_d[istart:iend+1]
        data = data[7:] #cut out some of the nonsense before recording starts
        idata = [x for x in data if len(x.split()) == 6]
        msgs  = [x for x in data if len(x.split()) != 6]
        
        idata = np.asarray([x.split() for x in idata])[:,:-1] #drop the last column as it is nonsense
        #idata now has: [trackertime, x, y, pupil, something random]
        if iseg == 0:
            fsamp = int(idata[0][0]) #get starting sample number
            eyedata.fsamp = fsamp
        #missing data is coded as '.' in the asc file - we need to make this usable but easily identifiable
        idata = np.where(idata=='.', 'NaN', idata) #set this missing data to a string nan that numpy can handle easily
        idata = idata.copy().astype(float) #convert into numbers now
        
        segdata             = EyeHolder()
        #segdata.fsamp       = fsamp
        segdata.trackertime = idata[:,0]
        segdata.xpos        = idata[:,1]
        segdata.ypos        = idata[:,2]
        segdata.pupil       = idata[:,3]
        segdata.time        = np.subtract(segdata.trackertime, segdata.trackertime[0]) #time relative to the first sample
        
        #some parsing of triggers etc
        msgs = [x.split() for x in msgs]
        triggers = np.array([x for x in msgs if x[0] == 'MSG'])
        segdata.triggers.timestamp = triggers[:,1].astype(int)
        segdata.triggers.event_id  = triggers[:,2]
        
        eyedata.data.append(segdata)
    return eyedata


#adding functions that really just work on pupil data, but that's ok as it's all we are interested in for now!

def strip_plr(data, plrtrigger, pre_buffer = 3):
    if type(data) != rawEyes:
        raise Exception(f'type must be rawEyes, not {type(data)}')
        
    for iblock in range(data.nblocks):
        if plrtrigger in data.data[iblock].triggers.event_id:
            tmpdata = data.data[iblock]
            plrtrigs = np.where(tmpdata.triggers.event_id == plrtrigger)[0] #get indices of plr triggers
            ftrig = plrtrigs[-1]+1 #get the next trigger after the last PLR (start of the first trial of task)
            ftrig_time = tmpdata.triggers.timestamp[ftrig]
            ftrigtime_cropped = ftrig_time - (data.srate*pre_buffer)
            
            #find all timepoints that occur before this cropped timepoint
            delinds = np.squeeze(np.where(tmpdata.trackertime < ftrigtime_cropped))
            
            #remove data from relevant signals
            tmpdata.xpos = np.delete(tmpdata.xpos, delinds)
            tmpdata.ypos = np.delete(tmpdata.ypos, delinds)
            tmpdata.pupil = np.delete(tmpdata.pupil, delinds)
            tmpdata.trackertime = np.delete(tmpdata.trackertime, delinds)
            tmpdata.fsamp = tmpdata.trackertime[0] #reset the first sample
            tmpdata.time = np.subtract(tmpdata.trackertime, tmpdata.fsamp) #update the time array
            
            trigs2rem = np.where(tmpdata.triggers.timestamp < ftrigtime_cropped)
            tmpdata.triggers.timestamp = np.delete(tmpdata.triggers.timestamp, trigs2rem)
            tmpdata.triggers.event_id  = np.delete(tmpdata.triggers.event_id, trigs2rem)
            
            #set the data
            data.data[iblock] = tmpdata
    
    #get the first sample again and update if needed
    fsamp = data.fsamp
    newfsamp = data.data[0].trackertime.min()
    if int(fsamp) <= int(newfsamp):
        data.fsamp = newfsamp
    
    return data #return the stripped data object


def interpolate_blinks(data):
    if type(data) != rawEyes:
        raise Exception(f'type must be rawEyes, not {type(data)}')
    
    for iblock in range(data.nblocks):
        tmpdata  = data.data[iblock]
        pupil    = tmpdata.pupil.copy()
        nanpupil = tmpdata.pupil_nan.copy()
        times    = tmpdata.time.copy()
        
        mask = np.zeros_like(times, dtype=bool)
        mask |= np.isnan(nanpupil)
        
        interpolated = np.interp(
            times[mask],
            times[~mask],
            pupil[~mask]
            )
        
        cleanpupil = nanpupil.copy()
        cleanpupil[mask] = interpolated
        data.data[iblock].pupil_clean = cleanpupil
    return data



