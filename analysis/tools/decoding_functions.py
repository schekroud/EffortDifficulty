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
    nclasses = np.unique(labels).size
    X = data[:,:,tp].copy()
    
    estimators = [('scaler', skl.preprocessing.StandardScaler())]
    
    if use_pca:
        estimators.append( ('reduce_dim', skl.decomposition.PCA(n_components = 0.95)) ) #add pca to the pipeline
    
    if classifier == 'ridge':
        estimators.append( ('clf', skl.linear_model.RidgeClassifier(fit_intercept=True) ) )
    if classifier == 'LDA':
        estimators.append( ('clf', skl.discriminant_analysis.LinearDiscriminantAnalysis() ) )
    if classifier == 'svm':
        estimators.append( ('clf', skl.svm.LinearSVC(dual = 'auto', random_state=420) ) )
    if classifier == 'knn':
        estimators.append( ('clf', skl.neighbors.KNeighborsClassifier(metric = 'mahalanobis',
                                                 metric_params={'V' :  np.cov(X.T), 'VI': np.linalg.inv(np.cov(X.T))} ) ) )
        
    clf = skl.pipeline.Pipeline(estimators)
    tmp = np.zeros_like(labels)

    
    if testtype == 'rskf':
        cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = nsplits, n_repeats=1)
    elif testtype == 'leaveoneout':
        cv = skl.model_selection.LeaveOneOut()
    
    for train_index, test_index in cv.split(X, labels):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        
    if classifier == 'ridge':
        decision = clf.decision_function(x_test) #shape = [nTestindex x classes]
        ps = np.zeros_like(decision) * np.nan
        for idist in range(decision.shape[0]): #loop over trials in the test set
            idec = decision[idist].reshape(1, -1)
            proba = np.squeeze(softmax(idec))
            ps[idist] = proba
        predproba = ps
    else:
        predproba = clf.predict_proba(x_test)
    
        tmp[test_index] = preds    
    
    return tmp #return label predictions for each trial, at this time point

def run_decoding_predproba(data, labels, tp, use_pca, testtype, classifier,  nsplits = 5):
    nclasses = np.unique(labels).size
    X = data[:,:,tp].copy()
    
    [ntrials, nfeatures]  = X.shape
    
    estimators = [('scaler', skl.preprocessing.StandardScaler())]
    
    if use_pca:
        estimators.append( ('reduce_dim', skl.decomposition.PCA(n_components = 0.95)) ) #add pca to the pipeline
    
    if classifier == 'ridge':
        estimators.append( ('clf', skl.linear_model.RidgeClassifier(fit_intercept=True) ) )
    if classifier == 'LDA':
        estimators.append( ('clf', skl.discriminant_analysis.LinearDiscriminantAnalysis() ) )
    if classifier == 'svm':
        estimators.append( ('clf', skl.svm.LinearSVC(dual = 'auto', random_state=420) ) )
    if classifier == 'knn':
        estimators.append( ('clf', skl.neighbors.KNeighborsClassifier() ) )#metric = 'mahalanobis',
                                                 #metric_params={'V' :  np.cov(X.T), 'VI': np.linalg.inv(np.cov(X.T))} ) ) )
        
    pipe = skl.pipeline.Pipeline(estimators)
    
    tmp = np.zeros_like(labels)
    predprobas = np.zeros(shape = [ntrials, nclasses])
    
    if testtype == 'rskf':
        cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = nsplits, n_repeats=1)
    elif testtype == 'leaveoneout':
        cv = skl.model_selection.LeaveOneOut()
    
    for train_index, test_index in cv.split(X, labels):
        # print('hi')
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if testtype == 'leaveoneout':
            x_test = x_test.reshape(1, -1)
        
        if classifier != 'knn':            
            pipe.fit(x_train, y_train)
            preds = pipe.predict(x_test)
            tmp[test_index] = preds    
        
        if classifier == 'ridge':
            decision = pipe.decision_function(x_test) #shape = [nTestindex x classes]
            ps = np.zeros_like(decision) * np.nan
            for idist in range(decision.shape[0]): #loop over trials in the test set
                idec = decision[idist].reshape(1, -1)
                proba = np.squeeze(softmax(idec))
                ps[idist] = proba
            predproba = ps
        elif classifier in ['LDA', 'svm']:
            predproba = pipe.predict_proba(x_test)
            predprobas[test_index] = predproba
            
        elif classifier == 'knn':
            scaler = skl.preprocessing.StandardScaler()
            x_train = scaler.fit(x_train).transform(x_train)
            x_test  = scaler.fit(x_train).transform(x_test)
            
            clf = skl.neighbors.KNeighborsClassifier(metric = 'mahalanobis', n_neighbors = 16,
                                                     metric_params = {'VI':np.linalg.inv(np.cov(x_train.T))})
            clf.fit(x_train, y_train)
            tmp[test_index] = clf.predict(x_test)
            predprobas[test_index] = clf.predict_proba(x_test)
    
    return tmp, predprobas #return label predictions for each trial, at this time point