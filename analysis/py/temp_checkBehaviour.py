#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:39:26 2023

@author: sammi
"""
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os.path as op
import os

wd = '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

datdir = op.join(wd, 'data', 'datafiles')

sub = 3
subdatdir = op.join(datdir, 's%02d'%sub)

files = sorted(os.listdir(subdatdir))





df = pd.DataFrame()

for file in files:
    fname = op.join(subdatdir, file)
    tmpdf = pd.read_csv(fname)
    df = pd.concat([df, tmpdf])


fig = plt.figure()
ax = fig.add_subplot(111)
sns.lineplot(df, x = 'trlid', y = 'difficultyOri', ax = ax)


plt.figure()
sns.barplot(df.groupby('difficultyOri', as_index = False).agg({'PerceptDecCorrect':'mean'}), x = 'difficultyOri', y = 'PerceptDecCorrect')