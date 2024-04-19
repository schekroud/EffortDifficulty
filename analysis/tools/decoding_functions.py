# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:32:20 2024

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as skl
from sklearn import *
from sklearn.utils.extmath import softmax

def run_decoding(data, labels, tp, use_pca, testtype, classifier,  nsplits = 5):
    X = data[:,:,tp].copy()
    pca = skl.decomposition.PCA(n_components = 0.95) #pca decomposition for dimensionality reduction
    X_pca = pca.fit(X).transform(X)
    
    tmp = np.zeros_like(labels) * np.nan
    if testtype == 'leaveoneout':
        cv = skl.model_selection.LeaveOneOut()
    elif testtype == 'rskf':
        cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = nsplits, n_repeats=1, random_state = 420)
    
    for train_index, test_index in cv.split(X, labels):
        if use_pca:
            x_train, x_test = X_pca[train_index], X_pca[test_index]
        else:
            x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if testtype == 'leaveoneout':
            x_test = x_test.reshape(1, -1) #fix for leave one out analyses or cant standardise features
            
        scaler = skl.preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test  = scaler.transform(x_test)
        
        if classifier == 'ridge':
            clf = skl.linear_model.RidgeClassifier(fit_intercept=True)
        if classifier == 'LDA':
            clf = skl.discriminant_analysis.LinearDiscriminantAnalysis(solver = 'svd')
        
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        tmp[test_index] = preds
    
    return tmp #return label predictions for each trial, at this time point

def run_decoding_predproba(data, labels, tp, use_pca, testtype, classifier,  nsplits = 5):
    nclasses = np.unique(labels).size
    X = data[:,:,tp].copy()
    pca = skl.decomposition.PCA(n_components = 0.95) #pca decomposition for dimensionality reduction
    X_pca = pca.fit(X).transform(X)
    
    tmp = np.zeros_like(labels) * np.nan
    predprobas = np.zeros(shape = [len(labels), nclasses]) * np.nan
    if testtype == 'leaveoneout':
        cv = skl.model_selection.LeaveOneOut()
    elif testtype == 'rskf':
        cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = nsplits, n_repeats=1, random_state = 420)
    
    for train_index, test_index in cv.split(X, labels):
    # x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(X, labels, test_size = 0.2)
        if use_pca:
            x_train, x_test = X_pca[train_index], X_pca[test_index]
        else:
            x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if testtype == 'leaveoneout':
            x_test = x_test.reshape(1, -1) #fix for leave one out analyses or cant standardise features
            
        scaler = skl.preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test  = scaler.transform(x_test)
        
        if classifier == 'ridge':
            clf = skl.linear_model.RidgeClassifier(fit_intercept=True)
        if classifier == 'LDA':
            clf = skl.discriminant_analysis.LinearDiscriminantAnalysis(solver = 'svd')
        
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        
        if classifier == 'LDA':
            predproba = clf.predict_proba(x_test)
        
        if classifier == 'ridge':
            decision = clf.decision_function(x_test) #shape = [nTestindex x classes]
            ps = np.zeros_like(decision) * np.nan
            for idist in range(decision.shape[0]): #loop over trials in the test set
                idec = decision[idist].reshape(1, -1)
                proba = np.squeeze(softmax(idec))
                ps[idist] = proba
            predproba = ps
        
        
        tmp[test_index] = preds
        
        predprobas[test_index] = predproba
        
    
    return tmp, predprobas #return label predictions for each trial, at this time point