#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:03:44 2022

@author: sammi
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a series of functions designed to aid the analysis of eyetracking data
collected in the Brain and Cognition Lab.

to use this, import the script when you are loading in other packages in your script, e.g.:

import BCEyes

Author: Sammi Chekroud

"""
import numpy   as np
import scipy   as sp
from scipy import interpolate
from scipy import signal
import copy
import sys
import os
from copy import deepcopy
np.set_printoptions(suppress = True)


class DataError(Exception):
    '''
    Raised when there is an error in the datafile and expectations do not align with what is present
    '''
    def _init_(self, msg):
        self.msg = msg

class ArgumentError(Exception):
    """
    Raised when there is an error in the arguments used in a function here
    """
    def _init_(self, msg):
        self.msg = msg

def Eucdist(x1, y1, x2, y2):
    """
    calculate euclidian distance between two points

    formula: sqrt( (x2-x1)^2 + (y2-y1)^2 )

    """
    distance = np.sqrt( (x2-x1)**2 + (y2-y1)**2)
    return distance

def parse_eye_data(eye_fname, block_rec, trial_rec, nblocks, ntrials = None, binocular = True):
    """

    eye_fname    -- full path to a file to be parsed. this is used as a template to create the parsed data in pickle form

    block_rec    -- boolean as to whether you stopped/started recording blockwise in your task

    trial_rec    -- boolean as to whethre you stopped/started recording trialwise in your task

    nblocks      -- the number of blocks of task run in your experiment. This must be defined in all cases.

    ntrials      -- the number of trials in your data if you recorded the eyes with trialwise stop/start
                    if you recorded blockwise, then ntrials is not needed.

                    if you recorded trialwise, then the number of trials per block
                    is calculated and used to separate the data into defined blocks
    binocular    -- this is a boolean specifying if you made a binocular recording (True if yes, False if monocular)
    """
    if not os.path.exists(eye_fname): #check that the path to the file exists
        raise Exception('the filename: %s does not exist. Please check!' %eye_fname)

    if 'block_rec' not in locals() and 'trial_rec' not in locals():
        raise ArgumentError('Please indicate whether the recording was stopped/started on each trial or each block')

    if 'block_rec' in locals():
        if block_rec == True:
            trial_rec = False
    if 'trial_rec' in locals():
        if trial_rec == True:
            block_rec = False


    if 'block_rec' in locals():
        if block_rec == True and 'nblocks' not in locals():
            raise ArgumentError('Please specify how many blocks you expect in your data recording')

    if 'trial_rec' in locals():
        if trial_rec == True and 'ntrials' not in locals():
            raise ArgumentError('Please specify how many trials you expect in your data recording')
        elif trial_rec == True and 'ntrials' in locals() and ntrials == None:
            raise ArgumentError('Please specify a number of trials to expect')
        elif trial_rec == True and 'ntrials' in locals() and ntrials != None and 'nblocks' not in locals():
            raise ArgumentError('Please specify how many blocks of data you recorded')


    if block_rec:
        d = _parse_eye_data_blockwise(eye_fname, nblocks, binocular)
    else:
        d = _parse_eye_data_trialwise(eye_fname, nblocks, ntrials)#, binocular)

    return d #return the parsed data


def _parse_eye_data_blockwise(eye_fname, nblocks, binocular):
    """
    this function is called by parse_eye_data and will operate on data where
    the recording was stopped/started for each block of task data
    """

    d = open(eye_fname, 'r')
    raw_d = d.readlines()
    d.close()

    split_d = []
    for i in range(len(raw_d)):
        tmp = raw_d[i].split()
        split_d.append(tmp)
        
    if binocular:
        start_inds = [x for x in range(len(split_d)) if len(split_d[x]) == 6 and split_d[x][0] == 'START']
    else:
        start_inds = [x for x in range(len(split_d)) if len(split_d[x]) == 5 and split_d[x][0] == 'START']
    if len(start_inds) != nblocks:
        raise DataError('%d blocks are found in the data, not %d as has been input in nblocks' %(len(start_inds),nblocks))

    #get points where recording stopped
    end_inds   = [x for x in range(len(split_d)) if len(split_d[x]) == 7 and split_d[x][0] == 'END']

    if len(start_inds) != len(end_inds):
        raise DataError('the number of times the recording was started and stopped does not align. check problems with acquisition')

    #assign some empty lists to get filled with information
    if binocular == True:
        traces = ['lx', 'rx', 'ly', 'ry', 'lp', 'rp']
    elif binocular == False:
        traces = ['x', 'y', 'p']

    blocked_data = np.array([]);


    for istart in range(len(start_inds)):

        tmpdata = dict() #this will house the arrays that we fill with information
        tmpdata['trackertime'] = np.array([])
        for trace in traces:
            tmpdata[trace] = np.array([])

        tmpdata['Efix'] = np.array([]); tmpdata['Sfix'] = np.array([])
        tmpdata['Esac'] = np.array([]); tmpdata['Ssac'] = np.array([])
        tmpdata['Eblk'] = np.array([]); tmpdata['Sblk'] = np.array([])
        tmpdata['Msg']  = np.array([])

        start_line = start_inds[istart]
        end_line   = end_inds[istart]

        iblk = np.array(split_d[start_line:end_line]) #get only lines with info for this block

        iblk_event_inds, iblk_events     = [], []
        iblk_blink_inds, iblk_blink      = [], []
        iblk_fix_inds, iblk_fix          = [], []
        iblk_sac_inds, iblk_sac          = [], []
        iblk_input_inds                  = []
        for x,y in np.ndenumerate(iblk):
            if y[0] == 'MSG':
                iblk_event_inds.append(x[0]) # add line index where a message was sent to the eyetracker
                iblk_events.append(y)        # add the line itself to list
            elif y[0] in ['EFIX', 'SFIX']:
                iblk_fix_inds.append(x[0])   # add line index where fixation detected (SR research)
                iblk_fix.append(y)           # add fixation event structure to list
            elif y[0] in ['ESACC', 'SSACC']:
                iblk_sac_inds.append(x[0])   # add line index where saccade detected (SR research)
                iblk_sac.append(y)           # add saccade event structure to list
            elif y[0] in ['EBLINK', 'SBLINK']:
                iblk_blink_inds.append(x[0]) # add line index where blink detected (SR research)
                iblk_blink.append(y)         # add blink event structure to list
            elif y[0] == 'INPUT':
                iblk_input_inds.append(x[0])  # find where 'INPUT' is in data (sometimes appears, has no use...)

        #the block events should really be an M x 3 shape array (because ['MSG', timestamp, trigger]).
        # if this isnt the case, you can't coerce to a shaped array (will be an array of lists :( ))
        #so find where this fails, and remove that event (it's likely to be: ['MSG', time_stamp, '!MODE', 'RECORD' ...]) as this is another silly line from eyelink

        events_to_remove = [x for x in range(len(iblk_events)) if len(iblk_events[x]) != 3]
        if len(events_to_remove) > 1:
            print ('warning, there are multiple trigger lines that have more than 3 elements to the line, check the data?')
        if len(events_to_remove) != 0:
            iblk_events.pop(events_to_remove[0]) #remove the first instance of more than 3 elements to the trigger line. should now be able to coerce to array with shape Mx3
        iblk_events = np.array(iblk_events)  #coerce to array for easier manipulation later on


        #get all non-data line indices
        iblk_nondata    = sorted(iblk_blink_inds + iblk_sac_inds + iblk_fix_inds + iblk_event_inds + iblk_input_inds)

        iblk_data = np.delete(iblk, iblk_nondata) #remove these lines so all you have is the raw data

        iblk_data = iblk_data[6:] #remove first five lines as these are filler after the recording starts

        try:
            iblk_data = np.vstack(iblk_data)
        except ValueError:
            for item in range(len(iblk_data)):
                iblk_data[item] = iblk_data[item][:7] #make sure each item has only 7 strings. sometimes the eyetracker hasn't appended the random .... at the end
            iblk_data = np.vstack(iblk_data) #shape of this should be number of columns in the file (data)

        if binocular:
            iblk_data = iblk_data[:,:7]            # only take first 7 columns as these contain data of interest. last one is redundant (if exists)
            eyecols = [1,2,4,5] #leftx, lefty, rightx, right y col indices
            for col in eyecols:
                missing_inds = np.where(iblk_data[:,col] == '.') #find where data is missing in the gaze position, as probably a blink (or its missing as lost the eye)
                for i in missing_inds:
                    iblk_data[i,col] = np.NaN #replace missing data ('.') with NaN
                    iblk_data[i,3]   = np.NaN #replace left pupil as NaN (as in a blink)
                    iblk_data[i,6]   = np.NaN #replace right pupil as NaN (as in a blink)
        elif not binocular:
            iblk_data = iblk_data[:,:4]
            eyecols = [1,2,3]
            for col in eyecols:
                missing_inds = np.where(iblk_data[:,col] == '.') #missing data is stored as '.' and needs nanning to be able to convert to float
                for i in missing_inds:
                    iblk_data[i, col] = np.NaN
        iblk_data = iblk_data.astype(np.float) # convert data from string to floats for computations

        #for binocular data, the shape is:
        # columns: time stamp, left x, left y, left pupil, right x, right y, right pupil
        tmpiblkdata = dict()
        tmpiblkdata['iblk_trackertime'] = iblk_data[:,0]
        if binocular:
            tmpiblkdata['iblk_lx'] = iblk_data[:,1]
            tmpiblkdata['iblk_ly'] = iblk_data[:,2]
            tmpiblkdata['iblk_lp'] = iblk_data[:,3]
            tmpiblkdata['iblk_rx'] = iblk_data[:,4]
            tmpiblkdata['iblk_ry'] = iblk_data[:,5]
            tmpiblkdata['iblk_rp'] = iblk_data[:,6]
        elif not binocular:
            tmpiblkdata['iblk_x'] = iblk_data[:,1]
            tmpiblkdata['iblk_y'] = iblk_data[:,2]
            tmpiblkdata['iblk_p'] = iblk_data[:,3]


        # split Efix/Sfix and Esacc/Ssacc into separate lists, and make into arrays for easier manipulation later on
        iblk_efix = np.array([iblk_fix[x] for x in range(len(iblk_fix)) if
                     iblk_fix[x][0] == 'EFIX'])
        iblk_sfix = np.array([iblk_fix[x] for x in range(len(iblk_fix)) if
                     iblk_fix[x][0] == 'SFIX'])
        iblk_ssac = np.array([iblk_sac[x] for x in range(len(iblk_sac)) if
                     iblk_sac[x][0] == 'SSACC'])
        iblk_esac = np.array([iblk_sac[x] for x in range(len(iblk_sac)) if
                     iblk_sac[x][0] == 'ESACC'])
        iblk_sblk = np.array([iblk_blink[x] for x in range(len(iblk_blink)) if
                     iblk_blink[x][0] == 'SBLINK'])
        iblk_eblk = np.array([iblk_blink[x] for x in range(len(iblk_blink)) if
                     iblk_blink[x][0] == 'EBLINK'])

        #create tmpdata (the block structure) by adding in relevant information
        for trace in traces:
            tmpdata[trace] = tmpiblkdata['iblk_' + trace]

        tmpdata['trackertime'] = tmpiblkdata['iblk_trackertime']
        tmpdata['Efix'] = iblk_efix   #this should have 8  columns
        tmpdata['Sfix'] = iblk_sfix   #this should have 3  columns
        tmpdata['Esac'] = iblk_esac   #this should have 11 columns
        tmpdata['Ssac'] = iblk_ssac   #this should have 3  columns
        tmpdata['Eblk'] = iblk_eblk   #this should have 5  columns
        tmpdata['Sblk'] = iblk_sblk   #this should have 3  columns
        tmpdata['Msg']  = iblk_events #this should have 3  columns
        
        tmpdata['binocular'] = binocular #log whether this was binocular data or not

        #tmpdata now contains the information for that blocks data. now we just need to add this to the blocked_data object before returning it

        blocked_data = np.append(blocked_data, copy.deepcopy(tmpdata))
    return blocked_data

def _parse_eye_data_trialwise(eye_fname, nblocks, ntrials):

    """
    this function is called by parse_eye_data and will operate on data where the recording was stopped/started for each trial of task data

    """

    d = open(eye_fname, 'r')
    raw_d = d.readlines()
    d.close()

    split_d = []
    for i in range(len(raw_d)):
        tmp = raw_d[i].split()
        split_d.append(tmp)

    #get all lines where 'START'i s seen, as this marks the start of the recording
    start_inds = [x for x in range(len(split_d)) if len(split_d[x]) == 6 and split_d[x][0] == 'START']
    if len(start_inds) != ntrials:
        raise DataError('%d trials are found in the data, not %d as has been input in ntrials' %(len(start_inds),ntrials))

    #get points where recording stopped
    end_inds   = [x for x in range(len(split_d)) if len(split_d[x]) == 7 and split_d[x][0] == 'END']

    if len(start_inds) != len(end_inds):
        raise DataError('the number of times the recording was started and stopped does not align. check problems with acquisition')

    #assign some empty lists to get filled with information
    trackertime = []
    lx   = []; rx   = []
    ly   = []; ry   = []
    lp   = []; rp   = []


    Efix = []; Sfix = []
    Esac = []; Ssac = []
    Eblk = []; Sblk = []
    Msg  = []

    for i in range(len(start_inds)):
        start_line = start_inds[i]
        end_line   = end_inds[i]

        itrl = np.array(split_d[start_line:end_line])

        itrl_event_inds, itrl_events     = [], []
        itrl_blink_inds, itrl_blink      = [], []
        itrl_fix_inds, itrl_fix          = [], []
        itrl_sac_inds, itrl_sac          = [], []
        itrl_input_inds                  = []
        for x,y in np.ndenumerate(itrl):
            if y[0] == 'MSG':
                itrl_event_inds.append(x[0]) # add line index where a message was sent to the eyetracker
                itrl_events.append(y)        # add the line itself to list
            elif y[0] in ['EFIX', 'SFIX']:
                itrl_fix_inds.append(x[0])   # add line index where fixation detected (SR research)
                itrl_fix.append(y)           # add fixation event structure to list
            elif y[0] in ['ESACC', 'SSACC']:
                itrl_sac_inds.append(x[0])   # add line index where saccade detected (SR research)
                itrl_sac.append(y)           # add saccade event structure to list
            elif y[0] in ['EBLINK', 'SBLINK']:
                itrl_blink_inds.append(x[0]) # add line index where blink detected (SR research)
                itrl_blink.append(y)         # add blink event structure to list
            elif y[0] == 'INPUT':
               itrl_input_inds.append(x[0])  # find where 'INPUT' is in data (sometimes appears, has no use...)

        #get all non-data line indices
        itrl_nondata    = sorted(itrl_blink_inds + itrl_sac_inds + itrl_fix_inds + itrl_event_inds + itrl_input_inds)

        itrl_data = np.delete(itrl, itrl_nondata) #remove these lines so all you have is the raw data

        itrl_data = itrl_data[6:] #remove first five lines as these are filler after the recording starts

        try:
            itrl_data = np.vstack(itrl_data) #if this fails its most likely because the end string of dots is missing for some reason
        except ValueError: #if fails, fix
            for item in range(len(itrl_data)):
                itrl_data[item] = itrl_data[item][:7] #make sure each item has only 7 strings. sometimes the eyetracker hasn't appended the random .... at the end

        itrl_data = itrl_data[:,:7]            # only take first 6 columns as these contain data of interest (redundant if above section trips up

        itrl_data = np.vstack(itrl_data) #shape of this should be number of columns in the file (data)

        #before you can convert to float, need to replace missing data where its '.' as nans (in pupil this is '0.0')
        eyecols = [1,2,4,5] #leftx, lefty, rightx, right y col indices
        for col in eyecols:
            missing_inds = np.where(itrl_data[:,col] == '.') #find where data is missing in the gaze position, as probably a blink (or its missing as lost the eye)
            for i in missing_inds:
                itrl_data[i,col] = np.NaN #replace missing data ('.') with NaN
                itrl_data[i,3]   = np.NaN #replace left pupil as NaN (as in a blink)
                itrl_data[i,6]   = np.NaN #replace right pupil as NaN (as in a blink)
        itrl_data = itrl_data.astype(np.float) #convert data from string to floats for computations

        #for binocular data, the shape is:
        # columns: time stamp, left x, left y, left pupil, right x, right y, right pupil
        itrl_trackertime = itrl_data[:,0]
        itrl_lx, itrl_ly, itrl_lp = itrl_data[:,1], itrl_data[:,2], itrl_data[:,3]
        itrl_rx, itrl_ry, itrl_rp = itrl_data[:,4], itrl_data[:,5], itrl_data[:,6]

        # split Efix/Sfix and Esacc/Ssacc into separate lists
        itrl_efix = [itrl_fix[x] for x in range(len(itrl_fix)) if
                     itrl_fix[x][0] == 'EFIX']
        itrl_sfix = [itrl_fix[x] for x in range(len(itrl_fix)) if
                     itrl_fix[x][0] == 'SFIX']

        itrl_ssac = [itrl_sac[x] for x in range(len(itrl_sac)) if
                     itrl_sac[x][0] == 'SSACC']
        itrl_esac = [itrl_sac[x] for x in range(len(itrl_sac)) if
                     itrl_sac[x][0] == 'ESACC']

        itrl_sblk = [itrl_blink[x] for x in range(len(itrl_blink)) if
                     itrl_blink[x][0] == 'SBLINK']
        itrl_eblk = [itrl_blink[x] for x in range(len(itrl_blink)) if
                     itrl_blink[x][0] == 'EBLINK']

        #append to the collection of all data now
        trackertime.append(itrl_trackertime)
        lx.append(itrl_lx)
        ly.append(itrl_ly)
        rx.append(itrl_rx)
        ry.append(itrl_ry)
        lp.append(itrl_lp)
        rp.append(itrl_rp)
        Efix.append(itrl_efix)
        Sfix.append(itrl_sfix)
        Ssac.append(itrl_ssac)
        Esac.append(itrl_esac)
        Sblk.append(itrl_sblk)
        Eblk.append(itrl_eblk)
        Msg.append(itrl_events)

    trialsperblock = ntrials/nblocks
    rep_inds  = np.repeat(np.arange(nblocks),trialsperblock)
    #plt.hist(rep_inds, bins = nblocks) # all bars should be the same height now if its work? @ trials per block (80)

    iblock = {
    'trackertime': [],
    'lx'  : [], 'ly'  : [], 'lp'  : [],
    'rx'  : [], 'ry'  : [], 'rp'  : [],
    'Efix': [], 'Sfix': [],
    'Esac': [], 'Ssac': [],
    'Eblk': [], 'Sblk': [],
    'Msg' : []
    }
    dummy = copy.deepcopy(iblock)
    blocked_data = np.repeat(dummy, nblocks)

    for i in np.arange(nblocks):
        inds = np.squeeze(np.where(rep_inds == i))
        iblock_data = copy.deepcopy(iblock)
        for ind in inds: #add trialwise info into the blocked structure, continuous data rather than sectioned
            iblock_data['trackertime'].append(trackertime[ind])
            iblock_data['lx'].append(lx[ind])
            iblock_data['ly'].append(ly[ind])
            iblock_data['lp'].append(lp[ind])
            iblock_data['rx'].append(rx[ind])
            iblock_data['ry'].append(ry[ind])
            iblock_data['rp'].append(rp[ind])
            iblock_data['Efix'].append(Efix[ind])
            iblock_data['Sfix'].append(Sfix[ind])
            iblock_data['Esac'].append(Esac[ind])
            iblock_data['Ssac'].append(Ssac[ind])
            iblock_data['Eblk'].append(Eblk[ind])
            iblock_data['Sblk'].append(Sblk[ind])
            iblock_data['Msg'].append(Msg[ind])
            blocked_data[i] = iblock_data
    for block in range(len(blocked_data)): #concatenate trialwise signals into whole block traces to make artefact removal easier
        blocked_data[block]['trackertime'] = np.hstack(blocked_data[block]['trackertime'])
        blocked_data[block]['lx']          = np.hstack(blocked_data[block]['lx']         )
        blocked_data[block]['ly']          = np.hstack(blocked_data[block]['ly']         )
        blocked_data[block]['lp']          = np.hstack(blocked_data[block]['lp']         )
        blocked_data[block]['rx']          = np.hstack(blocked_data[block]['rx']         )
        blocked_data[block]['ry']          = np.hstack(blocked_data[block]['ry']         )
        blocked_data[block]['rp']          = np.hstack(blocked_data[block]['rp']         )
    return blocked_data


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



def replace_points_outside_screen(data, nblocks, screen_dim, adjust_pupils = True, remove_win = 10):
    '''
    if you provide this function with traces to look at, and the dimensions of your screen (x,y), it will look for timepoints where
    the gaze coordinates are outside the possible parameters (e.g. points where signal was lost and x/y values are outside the limits of the screen)

    these will be replaced with nans. if you then subsequently use the find_missing_periods function, these will be seen as missing chunks and interpolation can be used.

    this function will take a window around these points and assign them all to nan (as usually there is a huge chunk) of data that is distorted with this kind of problem


    data            -- the data object that you're using (consistent with other functions)
    nblocks         -- the number of blocks of data (or trials, if you stopped/started for every trial)
    traces_to_scan  -- a list of the traces to look for this kind of problem in
    adjust_pupils   -- a boolean for whether or not you also want to correct the pupil diameter signal according to points in x/y outside of the screen (i.e. nan them out) - default to true
    screen_dim      -- dimensions of the screen in list format of: [x_dim,y_dim] (e.g. [1920, 1080])
    remove_win      -- the time window that will get remove either side of a point that is outside the screen dimensions


    note: this is only really obvious in the x and y coordinates. In the pupil data you get from eyelinks, there are no real 'outlying values' as it's arbitrary.
    Instead, for each eye, we'll nan the points in time that (for that eye) were outside the screen dimensions.
    e.g. the timestamps for points in 'lx' or 'ly' that were outside the screen size will be used to set the pupil data to nan for that eye.
    -- This is the default. If you don't want to do this, then when calling the functions, use: adjust_pupils = False
    '''

    if not isinstance(data, np.ndarray):
        raise DataError('check the format of your data, an array of dictionaries for each block is expected')

    if nblocks != len(data):
        raise DataError('there are not as many blocks in the data as you think, check this!')
    
    #check if binocular or monocular data as this controls the traces we scan
    binocular = data[0]['binocular']
    if binocular:
        traces_to_scan = ['lx', 'ly', 'rx', 'ry']
    elif not binocular:
        traces_to_scan = ['x', 'y']

    count = 0
    for block in data:
        print( 'finding values outside of the screen in block %02d/%02d'%(count+1, len(data)))

        for trace in traces_to_scan: #check if the traces you want to use are actually in this dataset
            if trace not in block.keys():
                raise DataError('the signal %s is missing from your data. check spelling, or the traces you want to pass in the function' %trace)

        #create an empty dictionary to load vectors into
        traces = {}
        for trace in traces_to_scan:
            traces['s_out_'+trace] = []
            traces['e_out_'+trace] = []

        for trace in traces_to_scan:
            if 'x' in trace:        #looking at the x coordinates
                relev_dim   = screen_dim[0]
            elif 'y' in trace:
                relev_dim   = screen_dim[1]
            tmptrace        = block[trace]
            tmpmtrace       = np.array(np.logical_or(tmptrace > relev_dim, tmptrace < 0), dtype = int)
            tmpdtrace       = np.diff(tmpmtrace) #find the change points (+1 means goes to above the upper limit, -1 means goes from above limit to under limit)
            traces['s_out_' + trace] = np.squeeze(np.where(tmpdtrace ==  1)) #start of the patch outside of screen
            traces['e_out_' + trace] = np.squeeze(np.where(tmpdtrace == -1)) #end   of the patch outside of screen
            if tmpmtrace[0] == 1: #starts with samples out of the screen dimensions:
                traces['s_out_'+trace] = np.insert(traces['s_out_'+trace], 0, 0) #add indicator that the trace starts out of screen dimensions
                traces['s_out_'+trace] = np.squeeze(traces['s_out_'+trace]) #un-array this if only one value, otherwise causes issues
                
            if traces['s_out_'+trace].size != traces['e_out_'+trace].size:
                if traces['e_out_'+trace].size > traces['s_out_'+trace].size: #weird case where if the second value in the trace is a nan, then the diff trace will have a value of -1 when it returns to a normal value ...
                    if traces['e_out_'+trace].size == 1: #block recording starts with data outside screen dimensions, so no 'start sample
                        traces['s_out_'+trace] = np.array(0)
                    elif traces['e_out_'+trace][0] == 0:
                        traces['e_out_'+trace] = np.delete(traces['e_out_'+trace],0) #remove this random thing then
                elif traces['s_out_'+trace][-1] > traces['e_out_'+trace][-1]: #there's also a scenario where sth like a blink happens towards the end of the block, so there is a start but no end
                    #in this scenario, set the end to be the last sample of the block (i.e. blocklength.size)
                    traces['e_out_'+trace] = np.append(traces['e_out_'+trace], len(tmptrace))


            if traces['s_out_' + trace].size > 0:
                for i in range(traces['s_out_' + trace].size):
                    if traces['s_out_'+trace].size == 1: #only one patch of data outside of the screen dimensions
                        start = traces['s_out_' + trace].tolist()
                        end   = traces['e_out_' + trace].tolist()
                    else:
                        start = traces['s_out_' + trace][i]
                        if i < traces['s_out_' + trace].size: #check in range of missing periods in the data)
                            end = traces['e_out_' + trace][i]
                        elif i == traces['s_out_' + trace].size:
                            end = traces['e_out_' + trace][-1] #if its the last start point, get the last end point
                        else:
                            end = traces['e_out_' + trace][-1] #get the last end point
                    if end != len(tmptrace):
                        start, end = start - remove_win, end + remove_win
                    elif end == len(tmptrace):
                        start, end = start-remove_win, end
                    block[trace][start:end] = np.nan #replace the datapoints outside of screen dimensions with nans
                    #bear in mind that this is done implicitly and it doesn't output what parts of the data have been changed!
        count += 1
    return data

def nanblinkperiods_fromeyelink(data, nblocks, traces_to_scan, remove_win = 50):

    '''
    this function should, hopefully, get blink periods from the eyetracker output and nan these periods in the data
    because blinks are a weird shape, it will also remove a time window (remove_win) before and after blink start/end

    '''
    ds = deepcopy(data)

    for i in range(nblocks):
        block = deepcopy(ds[i])

        allblinks = block['Eblk']
        rblinks = np.array([x for x in allblinks if x[1] == 'R'])
        rblink_starts = rblinks[:,2].astype(int)
        rblink_ends   = rblinks[:,3].astype(int)

        lblinks = np.array([x for x in allblinks if x[1] == 'L'])
        lblink_starts = lblinks[:,2].astype(int)
        lblink_ends   = lblinks[:,3].astype(int)

        #we want to nan a period around the blinks because there are some distortions of the signal nearby too
        rblink_starts_adjusted = np.subtract(rblink_starts, remove_win)
        rblink_ends_adjusted   = np.add(rblink_ends, remove_win)
        lblink_starts_adjusted = np.subtract(lblink_starts, remove_win)
        lblink_ends_adjusted   = np.add(lblink_ends, remove_win)

        traces = {}
        for trace in traces_to_scan:
            traces[trace] = block[trace]

        rblink_sampstorem = []
        for x in range(rblink_starts_adjusted.size):
            samps = np.arange(rblink_starts_adjusted[x], rblink_ends_adjusted[x]+1)
            sampstorem = np.where(np.isin(block['trackertime'], samps))
            rblink_sampstorem.extend(sampstorem)
        rblink_sampstorem = np.concatenate(rblink_sampstorem) #flatten list of arrays into one array

        lblink_sampstorem = []
        for x in range(lblink_starts_adjusted.size):
            samps = np.arange(lblink_starts_adjusted[x], lblink_ends_adjusted[x]+1)
            sampstorem = np.where(np.isin(block['trackertime'], samps))
            lblink_sampstorem.extend(sampstorem)
        lblink_sampstorem = np.concatenate(lblink_sampstorem) #flatten list of arrays into one array


        #now we can nan the appropriate traces

        for trace in traces_to_scan:
            if 'l' in trace:
                tmptrace = deepcopy(traces[trace])
                tmptrace[lblink_sampstorem] = np.nan
                traces[trace+'_nans'] = tmptrace
            elif 'r' in trace:
                tmptrace = deepcopy(traces[trace])
                tmptrace[rblink_sampstorem] = np.nan
                traces[trace+'_nans'] = tmptrace

#        plt.plot(traces['ly'], lw = .75, color = '#3182bd')
#        plt.plot(traces['ly_nans'], lw = .75, color = '#feb24c')

        for trace in traces_to_scan:
            block[trace] = traces[trace+'_nans']

        block['rblink_removedsamps'] = rblink_sampstorem
        block['lblink_removedsamps'] = lblink_sampstorem
        ds[i] = block

    return ds

def check_nansatstart(data, blocks_to_ignore = [], traces_to_clean = ['lx', 'rx'], sampstointerp = 10):
    '''
    this needs to be run after bce.find_missing_periods as it will check the 'blink' periods and see if durations are negative (which indicates an issue)
    by default it will ignore no blocks at all
    '''
    ds = deepcopy(data)
    binocular = ds[0]['binocular']
    if binocular:
        traces_to_clean = ['lx', 'ly', 'lp', 'rx', 'ry', 'rp']
    elif not binocular:
        traces_to_clean = ['x', 'y', 'p']

    if len(blocks_to_ignore) > 0:
        ignored = True
        ignoreblocks = blocks_to_ignore
    else:
        ignored = False

    traces_blocks2clean = {} #this stores which blocks contain nans at the start of the recording
    for trace in traces_to_clean:
        traces_blocks2clean[trace] = []
    
    if binocular:
        left_traces  = [trace for trace in traces_to_clean if 'l' in trace]
        right_traces = [trace for trace in traces_to_clean if 'r' in trace]
    elif not binocular:
        traces = [trace for trace in traces_to_clean]
    for i in range(len(data)):
        if not ignored:
            firstblinks = {}
            if binocular:
                firstblinks['lx'] = ds[i]['Eblk_lx'][0]
                firstblinks['rx'] = ds[i]['Eblk_rx'][0]
    
                if firstblinks['lx'][5] < 0:
                    for trace in left_traces:
                        traces_blocks2clean[trace].append(i)
                if firstblinks['rx'][5] < 0:
                    for trace in right_traces:
                        traces_blocks2clean[trace].append(i)
            
            elif not binocular:
                
                firstblinks['x'] = ds[i]['Eblk_x'][0]
                if firstblinks['x'][5] < 0: #indicates missing values at start
                    for trace in traces:
                        traces_blocks2clean[trace].append(i)
                    
        elif ignored:
            firstblinks = {}
            if i not in ignoreblocks:
                if binocular:
                    firstblinks['lx'] = ds[i]['Eblk_lx'][0]
                    firstblinks['rx'] = ds[i]['Eblk_rx'][0]
    
                    if firstblinks['lx'][5] < 0:
                        for trace in left_traces:
                            traces_blocks2clean[trace].append(i)
                    if firstblinks['rx'][5] < 0:
                        for trace in right_traces:
                            traces_blocks2clean[trace].append(i)
                elif not binocular:
                    firstblinks['x'] = ds[i]['Eblk_x'][0]
                    if firstblinks['x'][5] < 0: #indicates missing value at start
                        for trace in traces:
                            traces_blocks2clean[trace].append(i)

    #fix the missing data at the start of the recording

    for trace in traces_to_clean: #loop over traces that need fixing
        for iblock in traces_blocks2clean[trace]: #loop over blocks that need fixing
            print('fixing '+ trace +' in block '+ str(iblock))
            itrace = ds[iblock][trace]
            nansamps = np.isnan(itrace) #mark nan samples as true
            first_notnanind = np.squeeze(np.where(nansamps==False))[0] #get index of first non-missing sample
            itrace[:first_notnanind] = np.round(np.nanmean(itrace[:first_notnanind+sampstointerp]), 1) #interpolate by assigning average of the first 10 non-nan samples at the start of the block
            data[iblock][trace] = itrace
            data[iblock]['startinterp_'+trace+'_samps'] = np.arange(0, first_notnanind+sampstointerp)

    return data


def snip_end_of_blocks(data, snip_amount = 1.5, srate = 1000):
    '''
    will trim down the block data to 1.5s after the final trigger of the data in each block
    this can fix some problems like with lp filtering where nans are in the trace but its a little while after the task already finished
    '''

    ds = deepcopy(data)
    binocular = data[0]['binocular']
    if binocular:
        traces_to_snip = ['lx', 'ly', 'lp', 'rx', 'ry', 'rp']
    elif not binocular:
        traces_to_snip = ['x', 'y', 'p']
    
    for i in range(len(ds)):
        tmpblock = deepcopy(ds[i])
        trackertime = tmpblock['trackertime']
        lastmsg = tmpblock['Msg'][-1]
        ttime_lastmsg = float(lastmsg[1])
        sigind_lastmsg = np.squeeze(np.where(trackertime == ttime_lastmsg))
        snipend = int(sigind_lastmsg + snip_amount*srate) #add 1.5s to the end of this
        for trace in traces_to_snip:
            tmpblock[trace] = tmpblock[trace][:snipend]
        tmpblock['trackertime'] = tmpblock['trackertime'][:snipend]
        ds[i] = tmpblock
    
    return ds

def remove_before_task_start(data, snip_amount = 1.5, srate = 1000):
    '''
    This function snips the eyetracker data to exclude sections that used a PLR routine (as this biases mean estimates of the pupil response)
    
    The PLR routine was done before the first task block, and at the end of the last task block.
    
    The trigger for the start of a trial was `trig150` so this function simply finds the first trig150 (trial start)
    and removes everything that is more than 1.5s before it
    '''
    ds = deepcopy(data)
    
    binocular = data[0]['binocular']
    if binocular:
        traces_to_snip = ['lx', 'ly', 'lp', 'rx', 'ry', 'rp']
    elif not binocular:
        traces_to_snip = ['x', 'y', 'p']
    
    #trim first block
    block = deepcopy(data[0])
    blocktrigs = np.array(block['Msg'])
    trackertime = block['trackertime']
    
    first_trialstart = np.squeeze(np.where(blocktrigs[:,2]=='trig150'))[0]
    first_trlstart = blocktrigs[first_trialstart]
    first_trlstart_ttime = first_trlstart[1].astype(int)
    
    snip_before = int(first_trlstart_ttime - snip_amount*srate) #start 1.5s before the first task trigger
    snip_before_indsamp = np.squeeze(np.where(trackertime==snip_before)).astype(int)
    
    for trace in traces_to_snip:
        block[trace] = block[trace][snip_before_indsamp:]
    block['trackertime'] = block['trackertime'][snip_before_indsamp:]
    
    #remove the PLR triggers too as those obviously don't matter
    
    PLRtrigs = np.isin(blocktrigs[:,2], 'trigPLR_ON')
    
    blocktrigs = blocktrigs[~PLRtrigs] #remove the PLR triggers
    block['Msg'] = blocktrigs
    
    #also need to remove other things (blink/fixation messages) that occur before the start of the task block
    #remove Efix events in before task period
    ttime = block['trackertime'][0].astype(int)
    block['Efix'] = block['Efix'][np.greater_equal(block['Efix'][:,2].astype(int),ttime),:]
    block['Sfix'] = block['Sfix'][np.greater_equal(block['Sfix'][:,2].astype(int),ttime),:]
    block['Esac'] = block['Esac'][np.greater_equal(block['Esac'][:,2].astype(int),ttime),:]
    block['Ssac'] = block['Ssac'][np.greater_equal(block['Ssac'][:,2].astype(int),ttime),:]
    block['Eblk'] = block['Eblk'][np.greater_equal(block['Eblk'][:,2].astype(int),ttime),:]
    block['Sblk'] = block['Sblk'][np.greater_equal(block['Sblk'][:,2].astype(int),ttime),:]

    
    ds[0] = block
    return ds
    
    

#to boxcar smooth -- v simple bit of code
#import scipy as sp
#import scipy.signal
#set up boxcar
#boxcar = sp.signal.windows.boxcar(50) #50 ms/sample boxcar smoothing
#smoothed_signal = np.convolve(boxcar/boxcar.sum(), signal, move = 'same')


def cleanblinks_usingpupil(data, nblocks, signals_to_clean = ['x', 'y'], eyes = ['left', 'right']):

    '''
    function to detect blinks in the data using the pupil trace rather than gaze coordinates
    -- physiological limits for pupil dilation apparently make it easier to detect blinks.

    info collected here can be used to interpolate blinks in the gaze data.


    '''
    binocular = data[0]['binocular']
    
    blinkspd        = 2.5                                   #speed above which data is remove around nan periods -- threshold
    maxvelthresh    = 30
    maxpupilsize    = 20000
    cleanms         = 100                                    #ms padding around the blink edges for removal
    if binocular:
        traces_to_scan  = ['lp', 'lx', 'ly', 'rp', 'rx', 'ry']  #the namings of possible traces
    elif not binocular:
        traces_to_scan = ['x', 'y', 'p']

    count = 1
    for block in data:
        print( 'working on block %02d/%02d'%(count, nblocks),)
        traces = {}
        for trace in traces_to_scan:
            traces[trace] = block[trace]
        
        if binocular:
            badsamps = {}
            if 'left' in eyes:
                badsamps['left'] = []
            if 'right' in eyes:
                badsamps['right'] = []
        elif not binocular:
            badsamps = [] #if only one eye, we only need one vector

        #loop over each eye in the data -- depending on the eye, get the appropriate pupil trace to use for identifying blink regions
        if binocular:
            for eye in eyes:
                if eye == 'left':
                    signal = block['lp']
                    print ('on the left eye,',)
                if eye == 'right':
                    signal = block['rp']
                    print ('and on the right eye.')
    
                vel         = np.diff(signal)                               #derivative of pupil diameter (velocity)
                speed       = np.abs(vel)                                   #absolute velocity
                smoothv     = smooth(vel,   twin = 8, method = 'boxcar')    #apply some smoothing to the velocity signal to remove tremor in signal
                smoothspd   = smooth(speed, twin = 8, method = 'boxcar')    #apply some smoothing to the speed (absolute of derivative of the trace)
    
                #find bad samples determined by breaching max vel threshold, and nan them
                badsamples = np.squeeze(np.where(np.logical_or(speed >= maxvelthresh, signal[1:] > maxpupilsize)))
                signal[badsamples] = np.nan
    
                #going to loop over each sample (possibly slow..) to remove short segments of data surrounded by nans
                lastnan = 0; lastwasfast = False
                for i in range(len(signal)):
                    if np.isnan(signal[i]):
                        if np.logical_and((i-lastnan) < cleanms, i-lastnan > 0):
                            signal[lastnan:i] = np.nan
                        lastnan = i; lastwasfast = False
    
                        m = 1
                        while ( i-m>0 and ~np.isnan(signal[i-m]) and ( smoothspd[i-m] > blinkspd or (i-m-1>0 and smoothspd[i-m-1]>blinkspd) or (i-m-2 > 0 and smoothspd[i-m-2]>blinkspd))):
                            signal[i-m] = np.nan
                            m+=1
                    else:
                        if i < len(smoothspd): #can't index beyond the last sample of the pupil speedd
                            if (((i-lastnan == 1) or lastwasfast == True) and (smoothspd[i]>blinkspd or (i+1<len(signal) and smoothspd[i+1]>blinkspd) or (i+2<len(signal) and smoothspd[i+2]>blinkspd))):
                                signal[i] = np.nan
                                lastwasfast = True
                        if i == len(smoothspd): #last possible value of the pupil change speed
                            if i-lastnan == 1 or lastwasfast == True:
                                signal[i:i+1] = np.nan
                        else:
                            lastwasfast = False
    
                #get the points of the signal that are bad in the pupil, and nan the appropriate points in the traces that are to be cleaned too (for consistency)
                blanks = np.squeeze(np.where(np.isnan(signal))) #get the nan periods of the data after these cleaning processes
                if eye == 'left':
                    for trace in signals_to_clean:
                        traceid = 'l'+trace
                        traces[traceid][blanks] = np.nan #set these points to nan
                    traces['lp'] = signal
                    badsamps['left'] = blanks
                if eye == 'right':
                    for trace in signals_to_clean:
                        traceid = 'r'+trace
                        traces[traceid][blanks] = np.nan #set these points to nan
                    traces['rp'] = signal
                    badsamps['right'] = blanks
        elif not binocular:
            #same process basically, but just doing it once (cba making a function that does this elegantly, or subfunctions right now)
            signal = deepcopy(block['p']) #get pupil data
            print('cleaning data based on pupil data\n')
            vel         = np.diff(signal)                               #derivative of pupil diameter (velocity)
            speed       = np.abs(vel)                                   #absolute velocity
            smoothv     = smooth(vel,   twin = 8, method = 'boxcar')    #apply some smoothing to the velocity signal to remove tremor in signal
            smoothspd   = smooth(speed, twin = 8, method = 'boxcar')    #apply some smoothing to the speed (absolute of derivative of the trace)

            #find bad samples determined by breaching max vel threshold, and nan them
            badsamples = np.squeeze(np.where(np.logical_or(speed >= maxvelthresh, signal[1:] > maxpupilsize)))
            signal[badsamples] = np.nan

            #going to loop over each sample (possibly slow..) to remove short segments of data surrounded by nans
            lastnan = 0; lastwasfast = False
            for i in range(len(signal)):
                if np.isnan(signal[i]):
                    if np.logical_and((i-lastnan) < cleanms, i-lastnan > 0):
                        signal[lastnan:i] = np.nan
                    lastnan = i; lastwasfast = False

                    m = 1
                    while ( i-m>0 and ~np.isnan(signal[i-m]) and ( smoothspd[i-m] > blinkspd or (i-m-1>0 and smoothspd[i-m-1]>blinkspd) or (i-m-2 > 0 and smoothspd[i-m-2]>blinkspd))):
                        signal[i-m] = np.nan
                        m+=1
                else:
                    if i < len(smoothspd): #can't index beyond the last sample of the pupil speedd
                        if (((i-lastnan == 1) or lastwasfast == True) and (smoothspd[i]>blinkspd or (i+1<len(signal) and smoothspd[i+1]>blinkspd) or (i+2<len(signal) and smoothspd[i+2]>blinkspd))):
                            signal[i] = np.nan
                            lastwasfast = True
                    if i == len(smoothspd): #last possible value of the pupil change speed
                        if i-lastnan == 1 or lastwasfast == True:
                            signal[i:i+1] = np.nan
                    else:
                        lastwasfast = False
            #this has nan'd bad periods (where the rate of change is high)
            #leaves some data as 0 -- where there was no pupil it doesnt nan it in the recording, it saves a zero
            #just replace these
            signal = np.where(signal == 0, np.nan, signal)
            
            
            #get the points of the signal that are bad in the pupil, and nan the appropriate points in the traces that are to be cleaned too (for consistency)
            blanks = np.squeeze(np.where(np.isnan(signal))) #get the nan periods of the data after these cleaning processes
            for trace in traces_to_scan:
                traces[trace][blanks] = np.nan #set the blank periods to nan in the data
            traces['p'] = signal
            badsamps = blanks
            
        for key in traces.keys():
            block[key] = traces[key]

        block['badsamps'] = badsamps
        count += 1
    return data


def find_missing_periods(data, nblocks):
    '''
    this function will find all sections of missing data. this typically relates to blinks
    but it will also find small patches where an eye has been dropped from the data.
    this just finds all patches, no matter the size, that are missing and outputs meaningful info into your data structure

    data argument expects a list, where each item in the list is a dictionary containing the data for a continuous block.

    binocular recorded data is expected here, where in each block the following keys are in the data dict:
        lx, ly, rx, ry.
    these correspond to the left and right gaze values. this function will iterate over each one to identify missing periods
    in each signal, and then output this information into new keys in your data:
    Eblk_lx, Eblk_ly, Eblk_rx, Eblk_ry


    traces_to_scan -- a list of the traces that you actually want to scan for blinks is needed here
                      this allows some flexibility in what you scan (e.g. if not at all interested in gaze, you can just find missing periods for the pupil data)


    example:
        data = find_missing_periods(data, nblocks = 1, traces_to_scan = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry'])
    '''

    if not isinstance(data, np.ndarray):
        raise Exception('check the format of your data. an array of dictionaries for each block is expected')

    if nblocks != len(data): #len(data) should give you the number of blocks in the data file
        raise Exception('there are not as many blocks in the data as you think. check this!')

    binocular = data[0]['binocular']
    if binocular:
        traces_to_scan = ['lx', 'ly', 'lp', 'rx', 'ry', 'rp']
    elif not binocular:
        traces_to_scan = ['x', 'y', 'p']


    for block in data: #iterate over each block of data

        for trace in traces_to_scan:
            if trace not in block.keys():
                raise Exception('the signal %s is missing from your data. check spelling, or check the traces you want to pass to the function!' %trace)

        #create empty vectors to hold start and end points of missing data in the traces specified in function call
        tmpdata = dict()
        for trace in traces_to_scan:
            tmpdata['s_' + trace] = []
            tmpdata['e_' + trace] = []

        #find the missing data in each gaze trace
        for trace in traces_to_scan:
            tmp_mtrace = np.array(np.isnan(block[trace]) == True, dtype = int) #array's of 1/0's if data is missing at a sample
            tmp_dtrace = np.diff(tmp_mtrace) #find the change-points. +1 means goes from present to absent, -1 from absent to present
            tmpdata['s_' + trace] = np.squeeze(np.where(tmp_dtrace ==  1)) #find where it starts to be missing
            tmpdata['e_' + trace] = np.squeeze(np.where(tmp_dtrace == -1))+1 #find the index of the last missing sample (the last sampl is a nan, so need to get the next)


        for trace in traces_to_scan:
            tmpdata['Eblk_' + trace] = []

        for trace in traces_to_scan:               # loop over all traces
            if tmpdata['s_'+trace].size == tmpdata['e_'+trace].size +1: #when there's one more start than end, its usually because the eye dropped for the remainder of a block
                tmpdata['e_'+trace] = np.append(tmpdata['e_'+trace], len(block[trace])-1)
                tmpdata['e_'+trace]= np.squeeze(tmpdata['e_'+trace])
            for i in range(tmpdata['s_' + trace].size):  # loop over every start of missing data
                if tmpdata['s_' + trace].size == 1: # if only one missing period (unlikely in blocked data, but common if you get trialwise recordings)
                    start = tmpdata['s_' + trace].tolist()
                    end   = tmpdata['e_' + trace].tolist()
                    if np.size(end) == 0: #no end sample for the single missing period because it goes until the end of the block ...
                        end = len(block[trace])-1
                else:
                    start   = tmpdata['s_' + trace][i]
                    if i    < tmpdata['e_' + trace].size: #check within the range of missing periods in the data
                        end = tmpdata['e_' + trace][i]
                    elif i == tmpdata['e_' + trace].size:
                        end = tmpdata['e_' + trace][-1]
                    else:
                        end = tmpdata['e_' + trace][-1] #get the last end point
                ttime_start = block['trackertime'][start]
                ttime_end   = block['trackertime'][end]
                dur = np.subtract(end,start)

                # now we'll make a blink event structure
                # blink_code, start (blocktime), end (blocktime), start (trackertime), end (trackertime), duration
                evnt = [trace + '_BLK', start, end, ttime_start, ttime_end, dur]
                tmpdata['Eblk_' + trace].append(evnt)

        #append these new structures to the dataset...
        for trace in traces_to_scan:
            block['Eblk_' + trace] = tmpdata['Eblk_' + trace]
    return data

def interpolateBlinks_Blocked(block, trace, method = 'linear'):
    """

    example call: block = interpolateBlinks_Blocked(block, trace = 'lx')

    - This function will interpolate the blinks in eyelink data that is in blocks (longer continuous segments)
    - The data that it will handle can either be segmented trials (due to trialwise stop/start of recording in the task) stitched back together
    - or just continuous block data, depending on the structure you give it.

    - If you use the script AttSacc_ParseData.py to parse the data, and are running AttSacc_CleanBlockedData.py, then the block structure should be suitable for this function.

    To interpolate over the blinks present in a trace, the following fields must be present within the data dictionary:
    - trackertime: the time series of the timepoints defined by the eyelink time (rather than trial time!)
    - the trace (e.g. block['lx'])
    - the list of events characterising missing periods of data in that trace (e.g. block['Eblk_lx'])

    these blink structures have the following format:
        ['event_code', start (blocktime), end (blocktime), start (trackertime), end (trackertime), duration]
        event codes e.g. 'lx_BLK'


    Missing periods of data will be removed in the following way:
    periods of missing data of under 10 samples will be linearly interpolated within a window of 10 samples either side of the start and end of the period

    """
    if not isinstance(block, dict):
        raise Exception('data supplied is not a dictionary')

    if not isinstance(trace, str):
        raise Exception('the trace indicator supplied is not a string. \n please input a string e.g. \'lx\'')

    if trace not in block.keys():
        raise Exception('the signal you want to clean does not exist in the data. check your trace labels in your data')

    signal = block[trace] #extract the signal that needs cleaning

    eventlabel = 'Eblk_%s'%trace

    if eventlabel not in block.keys():
        raise Exception('the missing period information for this signal is not in your data structure')

    blinks = np.array(block[eventlabel])       #get the desired blink structures
    if blinks.shape[0] != 0:
        blinks = blinks[:,1:]            #remove the first column as it's a string for the event code, not needed now
        blinks = blinks.astype(float).astype(int) # change the strings to integers. need to go via float or it fails.


        short_duration_inds  = np.where(np.in1d(blinks[:,4], range(21)))[0]    # find the blinks that are below 20 samples long
        medium_duration_inds = np.where(np.in1d(blinks[:,4], range(21,51)))[0] # find the blinks that are between 21 and 50 samples long
        long_duration_inds   = np.where(blinks[:,4] > 50)[0]                   # find blinks that are over 50 samples long

        short_blinks  = blinks[short_duration_inds,:]
        medium_blinks = blinks[medium_duration_inds,:]
        long_blinks   = blinks[long_duration_inds,:]

        #linear interpolate across these smaller periods before proceeding.
        for blink in short_blinks:
            start, end               = blink[0], blink[1]+1 #get start and end periods of the missing data
            to_interp                = np.arange(start, end) #get all time points to be interpolated over

            #set up linear interpolation
            inttime                  = np.array([start,end])
            inttrace                 = np.array([signal[start], signal[end]])
            fx_lin                   = sp.interpolate.interp1d(inttime, inttrace, kind = 'linear')

            interptrace              = fx_lin(to_interp)
            signal[to_interp] = interptrace
        for blink in medium_blinks:
            start, end               = blink[0], blink[1] #get start and end periods of the missing data
            to_interp                = np.arange(start, end) #get all time points to be interpolated over

            #set up linear interpolation
            inttime                  = np.array([start,end])
            inttrace                 = np.array([signal[start], signal[end]])
            fx_lin                   = sp.interpolate.interp1d(inttime, inttrace, kind = 'linear')

            interptrace              = fx_lin(to_interp)
            signal[to_interp] = interptrace

        #now cubic spline interpolate across the larger missing periods (blinks)
        for blink in long_blinks:
            start, end            = blink[0], blink[1] #get start and end of these missing samples
            if end+40 >= signal.size: #this blink happens just before the end of the block, so need to adjust the window
                window            = [start-100, start - 50, end, signal.size-1] #reduce the window size but still cubic spline
            elif end+40 <= signal.size and end+80 >= signal.size:
                window            = [start-100, start-50, end+50, signal.size-1]
            else:
                window            = [start-100, start-50, end+50, end+100] #set the window for the interpolation
            inttime               = np.array(window)
            if end + 50 >= signal.size:
                inttrace          = np.array([np.nanmedian(signal[start-100:start-50])                 , np.nanmedian(signal[start-50:start-1]),
                                              np.nanmedian(signal[end:int(np.floor((signal.size-1-end)/2))]), np.nanmedian(signal[int(np.ceil((signal.size-1-end)/2)):signal.size-1]) ])
            elif end+50 <= signal.size and end + 100 >= signal.size:
                inttrace          = np.array([np.nanmedian(signal[start-100:start-50])                 , np.nanmedian(signal[start-50:start-1]),
                                              np.nanmedian(signal[end:end+50]), np.nanmedian(signal[end+50:signal.size-1]) ])
            else:
                inttrace          = np.array([np.nanmedian(signal[start-100:start-50]), np.nanmedian(signal[start-50:start-1]), # by giving the nanmedian between these points,
                                              np.nanmedian(signal[end+1:end+50])     , np.nanmedian(signal[end+50:end+100]) ])  # points, it accounts for variance of the signal
            fx_cub                = sp.interpolate.interp1d(inttime, inttrace, kind = method)


            if end+50 >= signal.size:
                to_interp         = np.arange(start-50, signal.size-1)
            else:
                to_interp         = np.arange(start-50, end+50) #interpolate just outside the start of the missing period, for cases of large changes due to blinks
            interptrace           = fx_cub(to_interp)
            signal[to_interp]     = interptrace
    #output the data into the block structure
    block[trace] = signal
    return block

def transform_pupil(data):
    '''
    '''
    
    ds = deepcopy(data)
    binocular = data[0]['binocular']
    if binocular:
        traces_to_scan = ['lp', 'rp']
    elif not binocular:
        traces_to_scan = ['p']
    
    nblocks = len(ds)
    
    for iblock in range(nblocks):
        traces = dict()
        for trace in traces_to_scan:
            traces[trace] = deepcopy(ds[iblock])[trace]
            traces[trace+'_mean'] = np.nanmean(traces[trace])
            traces[trace+'_perc'] = np.multiply(np.divide(np.subtract(traces[trace], traces[trace+'_mean']), traces[trace+'_mean']),100)
            ds[iblock][trace+'_perc'] = traces[trace+'_perc']
    
    return ds



def lpfilter_trace(data, ignore_blocks, traces_to_scan, lp_cutoff, srate = 1000, order = 2):
    '''

    just a quick function to lowpass any traces you want. At the moment it just applies a butterworth filter using scipy functions and sp.signal.filtfilt to create a zero phase filter
    (this is mostly used for saccade analyses but can also provide some smoothing of the data by removing high frequency activity)

    data           -- your data structure
    traces_to_scan -- the traces you want to lowpass filter
    lp_cutoff      -- your low pass cutoff (60Hz is a good choice)
    srate          -- sample rate of your data acquisition, needed to calculate nyquist (default: 1000Hz )
    order          -- the order (steepness of freq cutoff) of the filter you want to use (default: 2)
    '''

    for iblock in range(len(data)):
        if iblock not in ignore_blocks:
            dat = deepcopy(data[iblock])
    
            traces = {}
            for trace in traces_to_scan:
                traces[trace] = dat[trace]
    
            b, a = sp.signal.butter(order, Wn = lp_cutoff, btype = 'lowpass', fs = srate)
    
            for trace in traces_to_scan:
                traces[trace+'_filt'] = sp.signal.filtfilt(b, a, x = traces[trace])
    
            for trace in traces_to_scan:
                dat[trace] = traces[trace+'_filt']
    
            data[iblock] = dat

    return data

def lpfilter_epochs(epoched_data, trace, lp_cutoff, srate = 1000, order = 2):
    
    ds = deepcopy(epoched_data[trace])
    filtered = deepcopy(ds)
    b, a = sp.signal.butter(order, Wn = lp_cutoff, btype = 'lowpass', fs = srate)
    for i in range(ds.shape[0]):
        filttrace = sp.signal.filtfilt(b, a, x = ds[i,:])
        filtered[i,:] = filttrace
    
    return filtered
    





def epoch(data, trigger_values, traces, twin = [-.5, +1], srate = 1000):
    """
    this function will epoch your eyetracking data given a specific trigger to find.

    data                    -- this expects a list of dictionaries (the preferred data type for all the functions here), where (in theory, for now), each element of the list is a block of data
    trigger_values           -- this expects a string (as triggers to the eyelink are strings). This function will look for this specific trigger, not other variants containing this trigger
    twin                    -- time window for your epoch. Will default to .5s before, 1s after trigger onset unless specified otherwiseself.
    traces                  -- specify the traces that you want to epoch (e.g. 'lx', 'ly', 'lp', 'rp', etc..)
    srate                   -- the sampling rate of the eyetracker. Defaults to 1000Hz

    output:
        this will output a dictionary. Each key within the dictionary will hold a ntriggers x time array of epoched dataself.
        e.g.: output.keys() will give ['lx', 'rx', 'ly', 'ry', 'lp', 'rp']
                output['lx'] will be ntrigs x time array of epochs.

        output['info'] will be a dictionary containing info that makes porting into MNE for visualisation/statistics slightly easier.
    """

    #check some of the input arguments to make sure we've got the right inputs here
    #if not isinstance(trigger_value, str):
    #    raise Exception('please provide a string for a trigger, not a %s'%type(trigger_value))
    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        raise Exception('an array/list of dictionaries is expected as the input data')
    if not isinstance(traces, list):
        raise Exception('a list of strings is expected for the signal traces you want to epoch')

    print('looking for %d triggers'%(len(trigger_values)))

    output_data = np.array([]) #create empty output data that gets returned at the end
    info        = dict()       # create empty dict that we'll populate with data information
    twin_srate  = np.multiply(twin, srate)

    all_triggers = []
    for iblock in range(len(data)):
        all_triggers.append(data[iblock]['Msg'])
    all_triggers = np.reshape(np.concatenate(all_triggers).ravel(), (-1,3))

    trigfinds = []
    for x,y in enumerate(all_triggers):
        if y[2] in trigger_values:
            trigfinds.append(x)
    print('found %d events in the data to look for out of %d triggers in total'%(len(trigfinds),all_triggers.shape[0] ))
    trigfind_vals = all_triggers[trigfinds] #get the actual triggers of each event that we're epoching over
    numtrigs = len(trigfinds) #number of triggers (epochs) being found (made)

    for iblock in range(len(data)): # loop over all blocks of data in the structure
        block = copy.deepcopy(data[iblock])
        trace_dict = dict()
        for trace in traces:
            trace_dict[trace] = [] #create dummy vars for each trace being operated on
        triggers       = copy.deepcopy(block['Msg'])                                         #get a copy of all the triggers in this block of data

        to_exclude = []
        for x,i in enumerate(trigger_values):
            if i not in triggers[:,2]:
                print('%s not in block %d so excluding from epoching'%(i,iblock+1))
                to_exclude.append(i);

        #get triggers that we're looking for, and where they are in the data
        triginds = []
        triginds_val = []
        for x,y in enumerate(triggers):
            if y[2] in trigger_values:
                triginds.append(x)
                triginds_val.append(y[2])


        evs_to_find = np.array(triggers[triginds])    #what are the triggers we want to epoch around
        epoch_starts = evs_to_find[:,1].astype(float) #where do they start (in tracker time)
        epoch_starts_time = np.empty(shape=0)
        for starttime in epoch_starts:                #get where they start in time within signal, not trackertime
            epoch_starts_time = np.append(epoch_starts_time,int(np.squeeze(np.where(block['trackertime']==starttime))))

        #epoch around these triggers and add them to the trace dictionary
        for trace in traces:
            for trigstart in epoch_starts_time:
                tmpepoch = block[trace][int(trigstart)+int(twin_srate[0]):int(trigstart)+int(twin_srate[1])]
                trace_dict[trace].append(tmpepoch)
        output_data = np.append(output_data,trace_dict) #add each blocks trace dictionary (i.e. epochs for the traces) into output data

    #take this output data (blocked) and add all epochs into one list (not stacked into matrix form yet)
    output = {}
    for trace in traces:
        output[trace]=[]
    for blk in range(len(output_data)):
        for trace in traces:
            output[trace].append(output_data[blk][trace])
    for trace in traces:
        output[trace] = np.array([x for y in output[trace] for x in y])
        #output[trace] = np.ravel(output[trace])

    #now loop over each trial epoch to find if any trials where we couldn't epoch to the full width of the time window
    #in some cases, it may not be possible to epoch fully around a trigger if its close to the end of a task.
    #in this scenario, to be able to collapse across blocks (using vstack) we need to pad these trials with nans
    #we should then output if a trial has been nanpadded as it may need to be excluded.

    trace   = traces[0] #we only need to do this for one trace because they should all contain same amount of data
    to_pad  = [] #list of indices of trials we need to nan-pad
    timelen = int(twin_srate[1]-twin_srate[0]) #get the shape that we *should* have
    empty   = np.empty(shape=[timelen]); empty.fill(np.NaN)
    for x,y in enumerate(output[trace]): #loop across epochs
        if y.shape[0] != timelen:        #check if the epoch is of the correct length
            to_pad.append(x)             #if not, get index of that trial so we can nanpad

    padding=False
    if len(to_pad) != 0: #report how many trials need nan-padding for epoching and collapsing across blocks to work
        print('unable to epoch all trials equally, %d/%d trials need nan-padding'%(len(to_pad),numtrigs))
        padding = True

    if padding:            
        for i in to_pad:
            for trace in traces:
                empty   = np.empty(shape=[timelen]); empty.fill(np.NaN)
                trl_to_pad = output[trace][i]
                empty[:trl_to_pad.size]=trl_to_pad
            output[trace][i] = empty
            print('trial padding done')

    #now we have nan-padded trials, we should be able to collapse them all into a matrix easily
    for trace in traces:
        output[trace] = np.vstack(output[trace]) #stack all trials on top of eachother (makes averaging easier later on)

    # populate the info structure
    info['srate'] = srate
    info['trigger'] = trigfind_vals[:,2] #this is an array of what the trigger was for an epoch
    info['tmin']  = twin[0]
    info['tmax']  = twin[1]
    info['ch_names'] = output.keys() #channel names are the names of the keys of the data dictionary (all will have trace_epoch)
    info['ch_types'] = ['misc'] * len(output.keys())

    output['info'] = info #assign this info structure to the data file!

    return output

def collapse_blocks(output_data, traces, twin_srate):
    """
    function to collapse blocked epoched data into one larger array of all blocks instead of epochs within blocks
    needs to know what 'traces' are being used here, but inherits from the epoching function proper
    """

    blocknum = len(output_data) #get the number of blocks
    tmpstruct = dict()

    for trace in traces:
        tmpstruct[trace] = np.empty([0,int(twin_srate[1]- twin_srate[0])]) #make an empty array of the right shape so we can just vstack all the arrays easily

    for iblock in range(len(output_data)): #loop over all blocks
        for trace in traces: #loop over traces, to make sure we concatenate all epochs made
            tmpstruct[trace] = np.vstack([tmpstruct[trace], output_data[iblock][trace]]) # add the new blocks' epochs as new rows in the data

    return tmpstruct


def apply_baseline(epoched_data, traces, baseline_window, mode = 'mean',baseline_shift_gaze = None, binocular = None): #need to add binocular info earlier in the pipeline... (26/07/2018)
    """
    this will baseline to a traces that you specify in your epoched data

    epoched data            -- this function requires the data structure from the epoch function in this set of tools
    traces                  -- you need to specify the traces that you want to apply this to (e.g. ['rp', 'rx', 'ry', 'lp', 'lx', 'ly']
    baseline_window         -- specify the window you want to calculate your baseline with
    mode                    -- specify whether you want to mean or median baseline ('mean' or 'median').  NB uses nanmean/nanmedian just in case
    baseline_shift_gaze     -- if you're baselining to fixation (or a specific point), then provide a tuple/list of two numbers for x/y shifting
                               e.g. [960,540] for centre of a screen with dimensions 1920 x 1080
                               if None, then it doesn't add these back and leaves it de-meaned only
    binocular               -- this will eventually be a check for if binocular (as it will baseline only one trace rather than two)
    """

    info       = epoched_data['info']
    twin_srate = np.multiply([info['tmin'], info['tmax']],info['srate']).astype(int)
    timerange  = np.divide(np.arange(twin_srate[0], twin_srate[1], 1, dtype = float), info['srate'])
    bwin       = baseline_window #get the baseline window
    bwin_inds  = [int(np.squeeze(np.where(timerange == bwin[0]))), int(np.where(timerange == bwin[1])[0])] #start and end time of the baseline window in timerange

    for trace in traces:
        if mode       == 'mean':
            #get mean of the baseline window for each trial simultaneously as a vector
            bline_val = np.nanmean(epoched_data[trace][:,bwin_inds[0]:bwin_inds[1]],1)
            bline_val = bline_val.reshape(len(bline_val),1) #reshape so matrix subtractions work
        elif mode     == 'median':
            #get median of the baseline window for each trial simultaneously as a vector
            bline_val = np.nanmedian(epoched_data[trace][:,bwin_inds[0]:bwin_inds[1]],1)
            bline_val = bline_val.reshape(len(bline_val),1) #reshape so matrix subtractions work

        epoched_data[trace] = np.subtract(epoched_data[trace], bline_val)

        #if you specify values to baseline to...
        if baseline_shift_gaze != None: #we have an x,y to add to it...
            xshift = np.zeros((len(bline_val),1), dtype = float) #create trialnum x 1 array for addition
            yshift = np.zeros((len(bline_val),1), dtype = float)
            xshift.fill(baseline_shift_gaze[0])
            yshift.fill(baseline_shift_gaze[1])
            if 'x' in trace:
                epoched_data[trace] = np.add(epoched_data[trace], xshift)
            elif 'y' in trace:
                epoched_data[trace] = np.add(epoched_data[trace], yshift)

    return epoched_data #return the epoched data with traces now baselined


def average_eyes(epoched_data, traces):
    """
    function to average the left and right eyes (so only applicable if recorded binocular data...)

    epoched_data    -- expects data in the format from the epoching function here
    traces          -- which you want to average: e.g. traces = ['x', 'y', 'p']

    will output the same data structure with addition of new dict keys containing arrays of the averaged eyes for the traces

    """

    #some checks to see if things exist first...

    if 'pupil' in traces:
        if 'rp' and 'lp' not in epoched_data['info']['ch_names']:
            raise Exception('trying to average pupils but only have one pupil channel... check names/data!')
    if 'x' in traces:
        if 'lx' and 'rx' not in epoched_data['info']['ch_names']:
            raise Exception('trying to average x traces but only have one channel... check names/data')
    if 'y' in traces:
        if 'ly' and 'ry' not in epoched_data['info']['ch_names']:
            raise Exception('trying to average y traces but only have one channel... check names/data')

    #at this point now, should have two channels for x, y and pupil (that we can now average)

    if 'x' in traces:
        x = np.empty((2,epoched_data['lx'].shape[0], epoched_data['lx'].shape[1]))
        x[0,:,:], x[1,:,:] = epoched_data['lx'], epoched_data['rx']
        averaged_x = np.nanmean(x,0)
        epoched_data['ave_x'] = averaged_x

    if 'y' in traces:
        y = np.empty((2,epoched_data['ly'].shape[0], epoched_data['ly'].shape[1]))
        y[0,:,:], y[1,:,:] = epoched_data['ly'], epoched_data['ry']
        averaged_y = np.nanmean(y,0)
        epoched_data['ave_y'] = averaged_y

    if 'p' in traces:
        p = np.empty((2,epoched_data['lp'].shape[0], epoched_data['lp'].shape[1]))
        p[0,:,:], p[1,:,:] = epoched_data['lp'], epoched_data['rp']
        averaged_p = np.nanmean(p,0)
        epoched_data['ave_p'] = averaged_p
    
    if 'p_perc' in traces:
        p_perc = np.empty((2,epoched_data['lp_perc'].shape[0], epoched_data['lp_perc'].shape[1]))
        p_perc[0,:,:], p_perc[1,:,:] = epoched_data['lp_perc'], epoched_data['rp_perc']
        averaged_p_perc = np.nanmean(p_perc,0)
        epoched_data['ave_p_perc'] = averaged_p_perc

    return epoched_data

def average_eyes_blocked(data, traces):
    """
    function to average the left and right eyes (so only applicable if recorded binocular data...)

    blocked_data    -- expects data in the format from the parsing function here
    traces          -- which you want to average: e.g. traces = ['x', 'y', 'p']

    will output the same data structure with addition of new dict keys containing arrays of the averaged eyes for the traces

    """

    nblocks = len(data)

    if 'x' in traces:
        for i in range(nblocks):
            tmplx = data[i]['lx']
            tmprx = data[i]['rx']
            avex = np.nanmean(np.vstack([tmplx, tmprx]), axis=0) #nanmean across the eyes
            data[i]['ave_x'] = avex
    if 'y' in traces:
        for i in range(nblocks):
            tmply = data[i]['ly']
            tmpry = data[i]['ry']
            avey = np.nanmean(np.vstack([tmply, tmpry]), axis=0) #nanmean across the eyes
            data[i]['ave_y'] = avey
    if 'p' in traces:
        for i in range(nblocks):
            tmplp = data[i]['lp']
            tmprp = data[i]['rp']
            avep = np.nanmean(np.vstack([tmplp, tmprp]), axis=0) #nanmean across the eyes
            data[i]['ave_p'] = avep

    return data

def nanzscore(vector, zero_out_nans = True):
            '''
            zscore a vector ignoring nans
            optionally can set nans to 0 afterwards. useful for regressors
            '''
            vector = np.divide(np.subtract(vector, np.nanmean(vector)), np.nanstd(vector))
            if zero_out_nans:
                vector = np.where(np.isnan(vector), 0, vector)
            
            return vector


def drop_eyelink_messages(data):
    nblocks = len(data)
    for iblock in range(nblocks):
        tmpblock = deepcopy(data[iblock])
        del(tmpblock['Efix'], tmpblock['Sfix'], tmpblock['Esac'], tmpblock['Ssac'],
            tmpblock['Eblk'], tmpblock['Sblk'])
        data[iblock] = tmpblock
    
    return data


def nan_missingdata(data):
    '''
    notes:
        - this assumes monocular data just for ease
        - only works on pupil data as designed for pupillometry cleaning
    
    '''
    nblocks = len(data)
    
    for iblock in range(nblocks):
        tmpdata = deepcopy(data[iblock])
        tmpp = deepcopy(tmpdata['p'])
        tmpp = np.where(tmpp ==0, np.nan, tmpp)
        data[iblock]['p'] = tmpp #set this back
        
    return data

def removePeriodAroundBlinks(data, cleanbefore = 50, cleanafter = 100):
    '''
    notes:
        - this assumes monocular data just for ease
        - only works on pupil data as designed for pupillometry cleaning
    
    '''
    nblocks = len(data)
    
    for iblock in range(nblocks): #loop over blocks of data
            tmpdata = deepcopy(data[iblock])
            tmpp = deepcopy(tmpdata['p']) #get the pupil trace
            
            #find nans
            pupil_nans = np.isnan(tmpp).astype(int)
            nanchange = np.diff(pupil_nans)
            
            # fig = plt.figure(); ax = fig.add_subplot(111)
            # ax.plot(pupil_nans, color = 'b')
            # ax.plot(nanchange, color = 'r')
            
            nanstarts = np.where(nanchange == 1)[0] +1 #gets the index of the first nan in the period
            nanends = np.where(nanchange == -1)[0] +1  #this lets u index the last nan of the period
            
            nanstarts = np.subtract(nanstarts, cleanbefore) #adjusts to remove cleanms samples before the missing data
            nanends   = np.add(nanends, cleanafter)        #adjusts to remove cleanms samples after the missing data
            
            if len(nanstarts) == len(nanends): #blinks within the block not at start/end as these need fixing more effortfully
                for iblink in range(len(nanstarts)):
                    tmpp[nanstarts[iblink]:nanends[iblink]] = np.nan
                    
            #plot to visualise the effect of this step
            
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(data[iblock]['trackertime'], data[iblock]['p'], lw = 1, color = '#9ecae1', label = 'raw')
            ax.plot(tmpdata['trackertime'], tmpp, lw = 1, color = '#fdae6b', label = 'adjusted')
            ax.set_title('cleanms: %s'%str(cleanms))
                    
        


#def class(raweyes)
