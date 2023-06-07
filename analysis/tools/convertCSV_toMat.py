#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:03:21 2023

@author: sammi
"""
import os
import os.path as op
import numpy as np
import pandas as pd
import scipy as sp

wd =  '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
os.chdir(wd)

datdir = op.join(wd, 'data', 'datafiles')
files = os.listdir(datdir)
files = sorted([file for file in files if '.csv' in file])

for file in files:
    df = pd.read_csv(op.join(datdir, file))
    sp.io.savemat(op.join(datdir, file[0:-4]+'.mat'), {name: col.values for name, col in df.items()})