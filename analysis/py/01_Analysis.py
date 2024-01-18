#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:24:16 2024

@author: sammichekroud
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

wd = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty' #workstation wd
os.chdir(wd)

datapath = op.join(wd, 'data', 'datafiles')
df = pd.read_csv(op.join(datapath, 'EffortDifficulty_maindata_py.csv'))
#%%
subs = df.subid.unique()

#remove some subjects where needed
eegsubs = np.arange(10, subs.max()+1)
subs2use = [x for x in subs if x in eegsubs]

#problems with some participants overall:
# ppt 21 was falling asleep in the first block
# ppt 25 had the 'k' key blocked by tape in the first block, leading to problems with data in the block
# ppt 36 sleeping regularly in first three blocks, withdrew after three blocks of task
# ppt 37 falling asleep in blocks 2 & 3, fully asleep in large parts of block 3

subs2rem = [36, 37]
subs2use = np.array([x for x in subs2use if x not in subs2rem])

df = df.query('subid in @subs2use')
nsubs = df.subid.unique().size

#two participants had issues with the first block of their task - lets remove the first block for them
df = df.assign(to_remove = np.logical_and(np.isin(df.subid, [21, 25]), df.blocknumber == 1).astype(int)) #make filtering column
#remove block1 trials from these subjects, remove the unnecessary filtering column 
df = df.query('to_remove == 0').drop('to_remove', axis=1)
# these participants (21 and 25) will now have 256 trials, not 320 trials - this is expected

#%%


df_acc = df.query('fbgiven in ["correct", "incorrect"]').groupby( #first select only trials where they responded
    ['subid', 'trialdifficulty'], as_index=False).agg({'PerceptDecCorrect':'mean'})

plotacc = df_acc.groupby('trialdifficulty', as_index=False).agg({'PerceptDecCorrect':['mean', sp.stats.sem]})
plotacc.columns = ['difficulty', 'mean_acc', 'sem_acc']

fig = plt.figure(figsize = [5, 3])
ax = fig.add_subplot(111)
ax.bar(x = plotacc.difficulty+1,
       height = plotacc.mean_acc,
       yerr = plotacc.sem_acc,
       color = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'],
       width = 0.7)
ax.scatter(x = df_acc.trialdifficulty+1, y = df_acc.PerceptDecCorrect, color = 'k', s = 10)
ax.set_xticks([1, 2, 3, 4], labels = ['2','4','8','12'])
ax.set_xlabel('angle difference (°)')
ax.set_ylabel('mean accuracy')
ax.hlines(y = 0.5, lw = 2, ls = 'dashed', color = '#000000', xmin = 0.5, xmax =4.5)
ax.set_title('empirical performance')
fig.tight_layout()


df_rt = df.query('fbgiven in ["correct", "incorrect"]').groupby( #first select only trials where they responded
    ['subid', 'trialdifficulty'], as_index=False).agg({'resptime':'mean'})

plotrt = df_rt.groupby('trialdifficulty', as_index=False).agg({'resptime':['mean', sp.stats.sem]})
plotrt.columns = ['difficulty', 'mean_rt', 'sem_rt']

fig = plt.figure(figsize = [5, 3])
ax = fig.add_subplot(111)
ax.bar(x = plotrt.difficulty+1,
       height = plotrt.mean_rt,
       yerr = plotrt.sem_rt,
       color = ['#f1eef6','#bdc9e1','#74a9cf','#2b8cbe','#045a8d'],
       width = 0.7)
# ax.scatter(x = df_acc.trialdifficulty+1, y = df_rt.resptime, color = 'k', s = 10)
ax.set_xticks([1, 2, 3, 4], labels = ['2','4','8','12'])
ax.set_xlabel('angle difference (°)')
ax.set_ylabel('mean RT (s)')
ax.set_ylim([0.4, 0.6])
ax.set_title('empirical performance')
fig.tight_layout()

#%% do the random-effects approach to test for effect on accuracy

for isub in df.subid.unique(): #loop over subjects

    subdf = df.copy().query('subid == @isub')
    
    #set up the model (logistic regression)
    
    #drop timeout trials
    subdf = subdf.query('fbgiven != "timed out"') #this shouldn't throw out that many trials really
    
    intercept = np.ones(len(subdf)).astype(int) #intercept term
    diff      = subdf.difficultyOri.to_numpy().astype(int)
    diff2     = np.where(diff ==  2, 1, 0)
    diff4     = np.where(diff ==  4, 1, 0)
    diff8     = np.where(diff ==  8, 1, 0)
    diff12    = np.where(diff == 12, 1, 0)
    
    rt = subdf.rt.to_numpy()
    ISI = 
    
    
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
    
    
    
    
    logmodel = sm.discrete.discrete_model.Logit(endog, exog, kwargs)






