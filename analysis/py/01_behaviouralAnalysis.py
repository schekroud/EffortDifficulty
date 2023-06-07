#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:05:06 2023

@author: sammi
"""

import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import pickle
import statsmodels as sm
import statsmodels.api as sma
import seaborn as sns
%matplotlib

pd.set_option('display.max_columns', 500)

# sys.path.insert(0, '/Users/sammi/Desktop/postdoc/student_projects/Teresa/EffortDifficulty/tools')
# sys.path.insert(0, 'D:/Experiments/postdoc/student_projects/Teresa/analysis_scripts/tools')
#import eyefuncs as eyes
# import funcs as behfuncs

wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
# wd = 'D:/Experiments/postdoc/student_projects/Teresa'
os.chdir(wd)

datdir = op.join(wd, 'data', 'datafiles')


allsubs = np.arange(10+1)
subs = np.arange(3,10+1)

df = pd.DataFrame()

for sub in subs:
    fname = op.join(datdir, 'EffortDifficulty_s%02d_combined.csv'%sub)
    tmpdf = pd.read_csv(fname)
    df = pd.concat([df, tmpdf])
#df now contains all the data
#need to get whether a trial was a switch or stay trial (in the participants behaviour)
subids = df.subid.unique().tolist()

subs2use = [3, 4, 5, 6, 7, 8, 9, 10]
subs2use = [3, 4, 5, 6, 7,    9, 10] #4/5 had wrong difficulties, 8 was sleepy

df = df.query('subid in @subs2use')

#%%
dat_acc = pd.DataFrame()
lmfits = []
for sub in subs2use:
    tmpdf = deepcopy(df).query('subid == @sub')
    tmpdf = tmpdf.query('difficultyOri in [2, 4, 8, 12]')
    
    #logistic regression looking at accuracy 
    intercept = np.ones(len(tmpdf)).astype(int) #intercept (mean performance)
    anglediff = tmpdf.difficultyOri.to_numpy().astype(int)
    acc = tmpdf.PerceptDecCorrect.to_numpy().astype(int)
    acc = np.where(acc == -1, 0, acc)
    ISI = tmpdf.delay1.to_numpy()
    
    lm = sm.discrete.discrete_model.Logit(endog = acc,
                                          exog = np.array([intercept, anglediff, ISI]).T)
    lm_fit = lm.fit()
    #display(lm_fit.summary())
    lmfits.append(lm_fit) #add to a list of model fits per subject
    predicted = lm_fit.predict() #get predicted values
    
    
    tmpdf['predictedacc'] = predicted
    tmpdf['perceptacc'] = np.where(tmpdf.PerceptDecCorrect == -1, 0, tmpdf.PerceptDecCorrect)
    
    plot_schedules = False
    if plot_schedules:
        fig = plt.figure(figsize = (9,4))
        ax = fig.add_subplot(121)
        sns.barplot(ax = ax, data = tmpdf, palette = 'YlGnBu',
                    x = 'difficultyOri', y = 'predictedacc').set(xlabel = 'perceptual difficulty (°)', ylabel = 'mean fitted P(Correct)')
        ax.hlines(y = 0.5, lw = 1, ls = 'dashed', color = '#000000', xmin = -0.5, xmax = 4.5)
        ax=fig.add_subplot(122)
        sns.barplot(ax = ax, data = tmpdf, palette = 'YlGnBu',
                    x = 'difficultyOri', y = 'perceptacc').set(xlabel = 'perceptual difficulty (°)', ylabel = 'mean perceptual accuracy')
        ax.hlines(y = 0.5, lw = 1, ls = 'dashed', color = '#000000', xmin = -0.5, xmax = 4.5)
        fig.suptitle('subject ' + str(sub))
    dat_acc = pd.concat([dat_acc, tmpdf])

# plot these as group averages:    
datacc_gave = dat_acc.groupby(['subid', 'difficultyOri']).agg(
    {'predictedacc':'mean', 'perceptacc':'mean'}).reset_index(
        ).groupby('difficultyOri').agg(
            {'predictedacc':['mean', sp.stats.sem], 'perceptacc':['mean', sp.stats.sem]}).reset_index()
            
datacc_gave.columns = ['_'.join(col) for col in datacc_gave.columns.values]

fig_acc = plt.figure(figsize = (9,4))
ax = fig_acc.add_subplot(121)
ax.bar(x = datacc_gave.difficultyOri_,
       height = datacc_gave.perceptacc_mean,
       yerr = datacc_gave.perceptacc_sem,
       color = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'],
       width = 1)
ax.set_xticks([2, 4, 8, 12])
ax.set_xlabel('angle difference (°)')
ax.set_ylabel('mean accuracy')
ax.hlines(y = 0.5, lw = 1, ls = 'dashed', color = '#000000', xmin = -1, xmax = 13)
ax.set_title('empirical performance')

ax = fig_acc.add_subplot(122)
ax.bar(x = datacc_gave.difficultyOri_,
       height = datacc_gave.predictedacc_mean,
       yerr = datacc_gave.predictedacc_sem,
       color = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'],
       width = 1)
ax.set_xticks([2, 4, 8, 12])
ax.set_xlabel('angle difference (°)')
ax.set_ylabel('mean estimated P(correct)')
ax.hlines(y = 0.5, lw = 1, ls = 'dashed', color = '#000000', xmin = -1, xmax = 13)
ax.set_title('modelled performance')
#%%

# plot just empirical performance, with single subjects as points too

#%% modelling reaction times by angular difficulty and trial ISI
dat_rt = pd.DataFrame()
lmfits_rt = []
for sub in subs2use:
    tmpdf = deepcopy(df).query('subid == @sub')
    tmpdf = tmpdf.query('difficultyOri in [2, 4, 8, 12]')
    
    #drop nan trials (no response)
    tmpdf['to_remove'] = 0
    tmpdf['to_remove'] = np.where(np.isnan(tmpdf['rt']), 1, 0) #set to 1 if being removed
    tmpdf = tmpdf.query('to_remove == 0') #only keep non-nan rt trials
    
    #logistic regression looking at accuracy 
    intercept = np.ones(len(tmpdf)).astype(int) #intercept (mean performance)
    anglediff = tmpdf.difficultyOri.to_numpy().astype(int)
    angle2  = np.where(anglediff == 2,  1, 0)
    angle4  = np.where(anglediff == 4,  1, 0)
    angle8  = np.where(anglediff == 8,  1, 0)
    angle12  = np.where(anglediff == 12,  1, 0)
    
    rt = tmpdf.rt.to_numpy() #probably wanna zscore this
    ISI = tmpdf.delay1.to_numpy()
    
    #modelling trial difficulty as separate regressors doesnt work for this really as it just models mean performance not linear effect
    # reg = sm.regression.linear_model.OLS(endog = rt,
    #                                      exog = np.array([intercept, angle0, angle3, angle6, angle9, angle12, ISI]).T)
    reg = sm.regression.linear_model.OLS(endog = rt, hasconst = True, #because we are supplying a constant (intercept) to model the mean
                                         exog = pd.DataFrame(np.array([intercept, anglediff, ISI]).T,
                                                             columns = ['intercept', 'anglediff', 'ISI']))
    
    reg1 = sm.regression.linear_model.OLS(endog = rt, hasconst = False, 
                                          exog = pd.DataFrame(np.array([anglediff, ISI]).T,
                                                              columns = ['anglediff', 'ISI']))
    
    reg2 = sm.regression.linear_model.OLS(endog = rt, hasconst = True,
                                          exog = pd.DataFrame(np.array([intercept, angle2, angle4, angle8, angle12, ISI]).T,
                                                              columns = ['intercept', 'angle2', 'angle4', 'angle8', 'angle12', 'ISI']))
    #this model has no intercept so doesn't regress away average performance
    #rt then is just a function of angle differenc and ISI
    
    reg_fit  = reg.fit()
    reg1_fit = reg1.fit()
    reg2_fit = reg2.fit()
    
    #reg_fit = reg.fit()
    #display(reg_fit.summary())
    lmfits_rt.append(reg_fit.summary())
    predicted = reg_fit.predict()
    predicted1 = reg1_fit.predict()
    predicted2 = reg2_fit.predict()
    
    #tmpdf['predictedrt'] = predicted
    tmpdf = tmpdf.assign(predictedrt = predicted,
                         predicted1rt = predicted1,
                         predicted2rt = predicted2)
    
    plot_singlesub = False
    if plot_singlesub:
        fig = plt.figure(figsize = (9,4))
        ax = fig.add_subplot(141)
        sns.barplot(data = tmpdf, x = 'difficultyOri', y = 'rt', palette = 'YlGnBu', ax = ax).set(
            xlabel = 'perceptual difficulty (°)', ylabel = 'rt (s)', title = 'empirical rt data', ylim = [0.3, 0.5])
        ax = fig.add_subplot(142)
        sns.barplot(data = tmpdf, x = 'difficultyOri', y = 'predictedrt', palette = 'YlGnBu', ax = ax).set(
            xlabel = 'perceptual difficulty(°)', ylabel = 'fitted rt (s)', title = 'modelled rt data', ylim = [0.3,0.5])
        ax = fig.add_subplot(143)
        sns.barplot(data = tmpdf, x = 'difficultyOri', y = 'predicted1rt', palette = 'YlGnBu', ax = ax).set(
            xlabel = 'perceptual difficulty(°)', ylabel = 'fitted rt (s)', title = 'modelled rt data (reg1)', ylim = [0.3,0.5])
        ax = fig.add_subplot(144)
        sns.barplot(data = tmpdf, x = 'difficultyOri', y = 'predicted2rt', palette = 'YlGnBu', ax = ax).set(
            xlabel = 'perceptual difficulty(°)', ylabel = 'fitted rt (s)', title = 'modelled rt data (reg2)', ylim = [0.3,0.5])
        fig.tight_layout()
    
    dat_rt = pd.concat([dat_rt, tmpdf])


datrt_gave = dat_rt.groupby(['subid', 'difficultyOri'],as_index=False).agg(
    {'predictedrt':'mean', 'rt':'mean'}).groupby('difficultyOri', as_index=False).agg(
        {'predictedrt':['mean', sp.stats.sem], 'rt':['mean', sp.stats.sem]})
            
datrt_gave.columns = ['_'.join(col) for col in datrt_gave.columns.values]

fig = plt.figure(figsize = (9,4))
ax = fig.add_subplot(121)
ax.bar(x = datrt_gave.difficultyOri_,
       height = datrt_gave.rt_mean,
       yerr = datrt_gave.rt_sem,
       color = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'],
       width = 2)
ax.set_xticks([2, 4, 8, 12])
ax.set_xlabel('angle difference (°)')
ax.set_ylabel('rt (s)')
ax.set_title('empirical performance')
ax.set_ylim([0.3, 0.5])

ax = fig.add_subplot(122)
ax.bar(x = datrt_gave.difficultyOri_,
       height = datrt_gave.predictedrt_mean,
       yerr = datrt_gave.predictedrt_sem,
       color = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'],
       width = 2)
ax.set_xticks([2, 4, 8, 12])
ax.set_xlabel('angle difference (°)')
ax.set_ylabel('modelled rt (s)')
ax.set_title('modelled performance')
ax.set_ylim([0.3, 0.5])
fig.tight_layout()
#%%

#1 - determine switch rationality
#2 - calculate trials since/until switch was made
# --- this basically says how many trials since a switch and how many trials until the next switch (as ppl prepare to switch over time)
df_switches = pd.DataFrame()
for sub in subids:
    tmpsub = dat.query("subid == @sub")
    tmpsub = behfuncs.subject_streaks(tmpsub)
    tmpsub = behfuncs.get_difficulty_sequences(tmpsub)
    tmpsub = behfuncs.get_choice_sequences(tmpsub)
    tmpsub['prevtrlfb'] = np.select([np.isin(tmpsub.prevtrlfb, [120, 123]),
                                     np.isin(tmpsub.prevtrlfb, [121, 124]),
                                     np.isin(tmpsub.prevtrlfb, [122])],
                                  ['correct', 'incorrect', 'timed out'],
                                  default = 'nan')
    df_switches = pd.concat([df_switches, tmpsub])

#set up switch prediction model
df_pswitch = pd.DataFrame()
switchfits = []
for sub in subids:
    #two ways of modelling this:
    '''
    1:
    probability that the next trial will be different to the current (i.e. switching next)
    given the current trial is unrewarded
     - if i am told i was wrong, what is the probability that i switch my choice on the next trial?
     
     2: 
     probability that the current trial is different to the previous (switching on this trial)
     given the previous trial was unrewarded
     - if i was told on the previous trial that i was wrong, what is the probability that i switch on this current trial?
     
     i think number 2 is the most sensible to do.
    '''
    #2nd way of modelling
    isub = pd.DataFrame()
    tmpsub = df_switches.query("subid == @sub") #single subject data
    nruns = tmpsub.blocknumber.unique().size
    nLags = 10
    for run in np.add(1,range(nruns)):
        rundat = tmpsub.copy().query('blocknumber == @run')
        rundat['rewarded'] = np.where(rundat.fbgiven == 'correct', 1, 0)
        rundat['notrewarded'] = np.equal(rundat.rewarded, 0).astype(int) #invert to get whether a trial was unrewarded or not
        
        #add things into the dataframe to help make the design matrix for this block of task
        for lag in np.add(1, range(nLags)):
            rundat['lag'+str(lag)+'notrewarded'] = rundat.notrewarded.shift(lag)
        
        for lag in np.add(1, range(nLags)):
            rundat['lag'+str(lag)+'difficulty'] = rundat.difficultyOri.shift(lag)
            #difficulty is [0 12] degrees. Make it span [0 2] so demeaning makes it span [-1 1]
            rundat['lag'+str(lag)+'difficulty'] = np.divide(rundat['lag'+str(lag)+'difficulty'],6)
        
        rundat = rundat.assign(difficultyOri = np.divide(rundat.difficultyOri, 6))
        switchnext = rundat.switchnext.to_numpy()
        rundat = rundat.filter(regex = 'difficultyOri|notrewarded|lag')
        
        #demean regressors within run
        rundat = rundat.subtract(np.nanmean(rundat.values, axis=0)) #subtract mean of each column to demean regressors
        rundat.insert(0, 'switchnext', value=switchnext) #so we dont demean the dependent variable by accident
        isub = pd.concat([isub, rundat])
    # plt.imshow(isub, aspect='auto')
    # plt.contourf(isub, aspect='auto')
    
    
    #run glm within subject
    
    switch = isub.pop('switchnext')
    desMat = isub.values
    reg_names = isub.columns
    
    #find and remove nans from both the design matrix and the switch vector
    naninds = np.greater(np.isnan(desMat).sum(axis=1),0) #finds any row in the design matrix where there's a nan
    switch = switch[~naninds]
    desMat = desMat[~naninds]
    # desmatdf = pd.DataFrame(desMat, columns = reg_names)
    
    #also drop these trials from the single subject dataframe so we can store predictions
    tmpsub = tmpsub.loc[~naninds,:]
    
    logm = sm.discrete.discrete_model.Logit(endog = switch, exog = desMat)
    logmfit = logm.fit()
    switchfits.append(logmfit)
    
    tmpsub = tmpsub.assign(pswitch = logmfit.predict())
    df_pswitch = pd.concat([df_pswitch, tmpsub])
    


subbetas = pd.DataFrame()
for fit in switchfits:
    #loop over models
    betas = fit.params
    tmpbetas = pd.DataFrame([betas.values], columns = reg_names)
    subbetas = pd.concat([subbetas, tmpbetas])

subbetas = subbetas.assign(subid = subids)

sub_betas = pd.melt(subbetas, id_vars = 'subid')
sub_betas.insert(2, 'lag', 0)
sub_betas.insert(3, 'type', 0)

sub_betas = sub_betas.assign(type = np.where(np.logical_or(sub_betas.variable.str.startswith('difficulty'), sub_betas.variable.str.endswith('difficulty')), 'difficulty', 'notrewarded'))
# sub_betas['lag'] = [ re.findall(r'\d+', x) if len(re.findall(r'\d+', x)) > 0 for x in sub_betas.variable.to_numpy().astype(str)]

sub_betas['lag'] = [re.findall(r'\d+', x)[0] if re.findall(r'\d+',x) != [] else 0 for x in sub_betas.variable.to_numpy()]
sub_betas['lag'] = sub_betas.lag.astype(int)



subbetas_notrewarded = sub_betas.query('type == "notrewarded"')
subbetas_difficulty  = sub_betas.query('type == "difficulty"')


plot_unreward = subbetas_notrewarded.groupby('lag', as_index = False).agg({
    'value':['mean', sp.stats.sem]
    })
plot_unreward.columns = ['lag', 'mean_beta', 'sem_beta']

plot_diff = subbetas_difficulty.groupby('lag', as_index = False).agg({
    'value':['mean', sp.stats.sem]
    })
plot_diff.columns = ['lag', 'mean_beta', 'sem_beta']

#%%
#plot betas for both showing effect of trial history reward/difficulty on switching
fig = plt.figure()
ax = fig.add_subplot(111)
sns.lineplot(data = subbetas_notrewarded, x = 'lag', y = 'value', errorbar= 'se', color = '#8856a7',ax=ax, label = 'unrewarded') ##8856a7   #2b8cbe
sns.lineplot(data = subbetas_difficulty,  x = 'lag', y = 'value', errorbar = 'se', color = '#2b8cbe', ax = ax, label = 'difficulty')
ax.set_xticks(ticks = np.arange(11), labels = np.concatenate([np.array(['current']), np.add(np.arange(10),1).astype(str)]))
ax.set_ylabel('beta weight (AU)')
ax.hlines([0], xmin=-0.1, xmax=10.1, lw = 1, ls = 'dashed', color = '#000000')
#%%

'''
aim:
    single-trial metric of probability to switch on the next trial that can be used as a regressor in feedback analyses
        - can then look to see if there is a feedback signature that reflects an update or trigger to switch sides based on the task

steps:

1 - switch ~ difficulty 
2 = switch | unrewarded ~ difficulty

'''
df_pswitch = pd.DataFrame()
log1fits = []
log2fits = []
for sub in subids:
    isub = df_switches.copy().query('subid == @sub')
    
    #demean difficutly across trials
    
    isub = isub.assign(difficulty = np.subtract(isub.difficultyOri, isub.difficultyOri.mean()))
    isub = isub.assign(fbgiven = np.where(isub.fbgiven == 'timed out', 'incorrect', isub.fbgiven))


    log1 = sm.discrete.discrete_model.Logit.from_formula(data = isub,
            formula = 'switchnext ~ C(difficultyOri, Treatment(reference=0)) + C(fbgiven, Treatment(reference = "correct"))')
    log1fit = log1.fit()
    log1pred = log1fit.predict()
    log1fits.append(log1fit)
    
    #store modelled p(switch) from logistic regression of all trials
    isub = isub.assign(log1pswitch = log1pred)
    
    isub_unreward = isub.copy().query('fbgiven == "incorrect"') #take only trials where they were given feedback saying they were incorrect
    
    unreward_itrls = isub_unreward.trialnumber.to_numpy()
    
    log2 = sm.discrete.discrete_model.Logit.from_formula(data = isub_unreward,
                                                         formula = 'switchnext ~ C(difficultyOri, Treatment(reference=0))')
    log2fit = log2.fit()
    log2pred = log2fit.predict()
    log2fits.append(log2fit)
    
    isub = isub.assign(log2pswitch = np.nan)
    
    #store modelled p(switch) from logistic regression of p(switch|unrewarded)
    isub.loc[np.squeeze(np.where(isub.fbgiven == 'incorrect')), 'log2pswitch'] = log2pred
    
    df_pswitch = pd.concat([df_pswitch, isub])
    del(log1fit, log1pred, log2fit, log2pred)
    
    
#%%

#look at proportion of trials that are switches given unrewarded

df_propSwitch = df_pswitch.copy().query('fbgiven == "incorrect"')

propSwitches = df_propSwitch.groupby(['subid', 'difficultyOri'], as_index = False).agg(
    {'switchnext':'mean'}).groupby('difficultyOri', as_index=False).agg(
        {'switchnext':['mean', sp.stats.sem]})
propSwitches.columns = ['difficultyOri', 'mean', 'sem']

fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(111)
ax.bar(x=propSwitches.difficultyOri,
       height = propSwitches['mean'],
       yerr = propSwitches['sem'],
       color = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'],
       width = 2.6)
ax.set_xticks(ticks = [0,3,6,9,12])
ax.set_xlabel('angle difference (°)')
ax.set_ylabel('proportion switch | unrewarded')



df_pSwitchGivUnrew = df_pswitch.copy().query('fbgiven == "incorrect"')
df_pSwitchGivUnrew = df_pSwitchGivUnrew.groupby(['subid', 'difficultyOri'], as_index = False).agg(
    {'log2pswitch':'mean'}).groupby('difficultyOri', as_index=False).agg(
        {'log2pswitch':['mean', sp.stats.sem]})
propSwitches.columns = ['difficultyOri', 'mean', 'sem']
