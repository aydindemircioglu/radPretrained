#!/usr/bin/python3

import copy
import cv2
import hashlib
import itertools
import json
import logging
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import random
import shutil
import socket
import sys
import tempfile
import time

from collections import OrderedDict
from datetime import datetime
from functools import partial
from glob import glob
from joblib import Parallel, delayed
from pprint import pprint
from matplotlib import pyplot
from typing import Dict, Any

from pymrmre import mrmr
from scipy.stats import kendalltau
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel
from skfeature.function.statistical_based import f_score, t_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from ITMO_FS.filters.univariate import anova

from parameters import *
from helpers import *


def getMLExperiments (experimentList, expParameters, sKey, inject = None):
    newList = []
    for exp in experimentList:
        for cmb in list(itertools.product(*expParameters.values())):
            pcmb = dict(zip(expParameters.keys(), cmb))
            if inject is not None:
                pcmb.update(inject)
            _exp = exp.copy()
            _exp.append((sKey, pcmb))
            newList.append(_exp)
    experimentList = newList.copy()
    return experimentList


# this is pretty non-generic, maybe there is a better way, for now it works.
def generateAllExperiments (experimentParameters, verbose = False):
    experimentList = [ [] ]
    for k in experimentParameters.keys():
        if verbose == True:
            print ("Adding", k)
        elif k == "FeatureSelection":
            # this is for each N too
            print ("Adding feature selection")
            newList = []
            for n in experimentParameters[k]["N"]:
                for m in experimentParameters[k]["Methods"]:
                    fmethod = experimentParameters[k]["Methods"][m].copy()
                    fmethod["nFeatures"] = [n]
                    newList.extend(getMLExperiments (experimentList, fmethod, m))
            experimentList = newList.copy()
        elif k == "Classification":
            newList = []
            for m in experimentParameters[k]["Methods"]:
                newList.extend(getMLExperiments (experimentList, experimentParameters[k]["Methods"][m], m))
            experimentList = newList.copy()
        else:
            experimentList = getMLExperiments (experimentList, experimentParameters[k], k)

    return experimentList



def preprocessData (X, y, simp = None, sscal = None):
    if simp is None:
        simp = SimpleImputer(strategy="mean")
        X = pd.DataFrame(simp.fit_transform(X),columns = X.columns)
    else:
        X = pd.DataFrame(simp.transform(X),columns = X.columns)

    if sscal is None:
        sscal = StandardScaler()
        X = pd.DataFrame(sscal.fit_transform(X),columns = X.columns)
    else:
        X = pd.DataFrame(sscal.transform(X),columns = X.columns)

    return X, y, simp, sscal



def applyFS (X, y, fExp):
    print ("Applying", fExp)
    return X, y



def applyCLF (X, y, cExp, fExp = None):
    print ("Training", cExp, "on FS:", fExp)
    return "model"



def testModel (y_pred, y_true, idx, fold = None):
    t = np.array(y_true)
    p = np.array(y_pred)

    # naive bayes can produce nan-- on ramella2018 it happens.
    # in that case we replace nans by 0
    p = np.nan_to_num(p)
    y_pred_int = [int(k>=0.5) for k in p]

    acc = accuracy_score(t, y_pred_int)
    df = pd.DataFrame ({"y_true": t, "y_pred": p}, index = idx)

    return {"y_pred": p, "y_test": t,
                "y_pred_int": y_pred_int,
                "idx": np.array(idx).tolist()}, df, acc




def getAUCCurve (modelStats, dpi = 100):
    # compute roc and auc
    fpr, tpr, thresholds = roc_curve (modelStats["y_test"], modelStats["y_pred"])
    area_under_curve = auc (fpr, tpr)
    if (math.isnan(area_under_curve) == True):
        print ("ERROR: Unable to compute AUC of ROC curve. NaN detected!")
        print (modelStats["y_test"])
        print (modelStats["y_pred"])
        raise Exception ("Unable to compute AUC")
    sens, spec = findOptimalCutoff (fpr, tpr, thresholds)

    if dpi > 0:
        f, ax = plt.subplots(figsize = (6,6), dpi = dpi)
        ax.plot(fpr, tpr, 'b')
        #plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_aspect('equal', 'datalim')
    else:
        f, ax = None, None

    return (f, ax), area_under_curve, sens, spec



def getPRCurve (modelStats, dpi = 100):
    # compute roc and auc
    precision, recall, thresholds = precision_recall_curve(modelStats["y_test"], modelStats["y_pred"])
    try:
        f1 = f1_score (modelStats["y_test"], modelStats["y_pred_int"])
    except Exception as e:
        print (modelStats["y_test"])
        print (modelStats["y_pred_int"])
        raise (e)
    f1_auc = auc (recall, precision)
    if (math.isnan(f1_auc) == True):
        print ("ERROR: Unable to compute AUC of PR curve. NaN detected!")
        print (modelStats["y_test"])
        print (modelStats["y_pred"])
        raise Exception ("Unable to compute AUC")


    if dpi > 0:
        f, ax = plt.subplots(figsize = (6,6), dpi = dpi)
        ax.plot(recall, precision, 'b')
        #plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.set_aspect('equal', 'datalim')
    else:
        f, ax = None, None

    return (f, ax), f1, f1_auc



def logMetrics (foldStats):
    #print ("Computing stats over all folds", len([f for f in foldStats if "fold" in f]))
    y_preds = []
    y_test = []
    y_index = []
    #pprint (foldStats)
    for k in foldStats:
        if "fold" in k:
            y_preds.extend(foldStats[k]["y_pred"])
            y_test.extend(foldStats[k]["y_test"])
            y_index.extend(foldStats[k]["idx"])

    modelStats, df, acc = testModel (y_preds, y_test, idx = y_index, fold = "ALL")
    (f_ROC, ax_ROC), roc_auc, sens, spec = getAUCCurve (modelStats, dpi = -1)
    (f_PR, ax_PR), f1, f1_auc = getPRCurve (modelStats, dpi = -1)

    expVersion = '_'.join([k for k in foldStats["params"] if "Experiment" not in k])
    pID = str(foldStats["params"])

    # register run in mlflow now
    foldStats["model"] = modelStats
    foldStats["Accuracy"] = acc
    foldStats["Sens"] = sens
    foldStats["Spec"] = spec
    foldStats["AUC"] = roc_auc
    foldStats["F1"] = f1
    foldStats["F1_AUC"] = f1_auc
    foldStats["preds"] = df
    print(".", end = '', flush=True)
    return foldStats




def kendall_corr_fct (X, y):
    scores = [0]*X.shape[1]
    for k in range(X.shape[1]):
        scores[k] = 1-kendalltau(X[:,k], y)[1]
    return np.array(scores)


def mrmre_score (X, y, nFeatures):
    Xp = pd.DataFrame(X, columns = range(X.shape[1]))
    yp = pd.DataFrame(y, columns=['Target'])

    # we need to pre-specify the max solution length...
    solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=5)
    scores = [0]*Xp.shape[1]
    for k in solutions.iloc[0]:
        for j, z in enumerate(k):
            scores[z] = scores[z] + Xp.shape[1] - j
    scores = np.asarray(scores, dtype = np.float32)
    scores = scores/np.sum(scores)
    return scores


def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = [0]*X.shape[1]
    for j in range(X.shape[1]):
        xn = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j] - np.min(X[:,j])))
        xn = xn/np.sum(xn)
        xn = np.asarray(xn, dtype = np.float32)
        scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    # ties = {i:list(scores).count(i) for i in scores if list(scores).count(i) > 1}
    # print(ties)
    return -scores



def createFSel (fExp):
    method = fExp[0][0]
    nFeatures = fExp[0][1]["nFeatures"]
    pipe = None

    if method == "LASSO":
        C = fExp[0][1]["C"]
        clf = LogisticRegression(penalty='l1', max_iter=500, solver='liblinear', C = C)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures)


    if method == "Anova":
        pipe = SelectKBest(anova, k = nFeatures)


    if method == "ET":
        clf = ExtraTreesClassifier(random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)


    if method == "RF":
        clf = RandomForestClassifier(random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)


    if method == "Bhattacharyya":
        pipe = SelectKBest(bhattacharyya_score_fct, k = nFeatures)


    if method == "tScore":
        def t_score_fct (X, y):
            scores = t_score.t_score (X,y)
            return scores
        pipe = SelectKBest(t_score_fct, k = nFeatures)


    if pipe is None:
        raise Exception ("Method", method, "is unknown")
    return pipe



def createClf (cExp, x_shape):
    #print (cExp)
    method = cExp[0][0]

    if method == "RBFSVM":
        C = cExp[0][1]["C"]
        g = cExp[0][1]["gamma"]
        model = SVC(kernel = "rbf", C = C, gamma = g, probability = True)

    if method == "LogisticRegression":
        C = cExp[0][1]["C"]
        model = LogisticRegression(solver = 'liblinear', C = C, random_state = 42)

    if method == "LinearSVM":
        alpha = cExp[0][1]["alpha"]
        model = SGDClassifier(alpha = alpha, loss = "log")

    if method == "RandomForest":
        n_estimators = cExp[0][1]["n_estimators"]
        model = RandomForestClassifier(n_estimators = n_estimators)

    if method == "NaiveBayes":
        model = GaussianNB()

    if method == "NeuralNetwork":
        N1 = cExp[0][1]["layer_1"]
        N2 = cExp[0][1]["layer_2"]
        N3 = cExp[0][1]["layer_3"]
        model = MLPClassifier (hidden_layer_sizes=(N1,N2,N3,), random_state=42, max_iter = 32)
    return model



@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
def executeExperiment (fselExperiments, clfExperiments, data, dataID, expID, eConfig):

    fResults = os.path.join(resultsPath, dataID + "_" + expID)
    os.makedirs (fResults, exist_ok = True)

    # we have that
    fResults = os.path.join(fResults,  dict_hash(eConfig) + ".dump")
    if os.path.exists(fResults):
        print ("X", end = '', flush = True)
        return None

    X = data.copy()

    # augmentations can interfere with stratification, ensure it is not
    baseX = X.groupby('Patient').first().reset_index()
    baseY = baseX["Target"]

    # extract exp
    assert (len(fselExperiments) == 1)
    assert (len(clfExperiments) == 1)

    fExp = fselExperiments[0]
    cExp = clfExperiments[0]

    # we only have one experiment
    stats = {}
    np.random.seed(42)
    random.seed(42)

    stats = {}
    stats["features"] = []
    stats["N"] = X.shape[0]
    stats["params"] = {}
    stats["params"].update(fExp)
    stats["params"].update(cExp)

    # need a fixed set of folds to be comparable
    timeFSStart = time.time()

    kfolds = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 1, random_state = 42)
    for k, (train_index, test_index) in enumerate(kfolds.split(baseX, baseY)):
        trainIDs = list(baseX.iloc[train_index]["Patient"])
        testIDs = list(baseX.iloc[test_index]["Patient"])

        if len(set(trainIDs).intersection(set(testIDs))) != 0:
            raise Exception ("Splitting did not work!")

        # now retrieve the data from the original dataset, but only for train
        X_train = X.query('Patient in @trainIDs').copy()
        y_train = X_train["Target"]

        # for test we use test-time-augmentations
        # right now we could not identify the one without augmentation anyway.
        X_test = X.query('Patient in @testIDs').copy()
        y_test = X_test["Target"].copy()
        testIDs = X_test["Patient"]

        X_train = X_train.drop(["Target", "Patient"], axis = 1)
        X_test = X_test.drop(["Target", "Patient"], axis = 1)

        # make sure we have something numeric, at least for mrmre
        y_train = y_train.astype(np.uint8)

        # scale
        X_train, y_train, simp, sscal = preprocessData (X_train, y_train)
        X_test, y_test, _, _ = preprocessData (X_test, y_test, simp, sscal)

        # create fsel
        fselector = createFSel (fExp)
        with np.errstate(divide='ignore',invalid='ignore'):
            fselector.fit (X_train.copy(), y_train.copy())
        feature_idx = fselector.get_support()
        feature_names = X_train.columns[feature_idx].copy()
        stats["features"].append(list([feature_names][0].values))

        # apply selector-- now the data is numpy, not pandas, lost its names
        X_fs_train = fselector.transform (X_train)
        y_fs_train = y_train

        X_fs_test = fselector.transform (X_test)
        y_fs_test = y_test

        # check if we have any features
        if X_fs_train.shape[1] > 0:
            classifier = createClf (cExp, X_fs_train.shape)
            classifier.fit (np.array(X_fs_train, dtype = np.float32), np.array(y_fs_train, dtype = np.int64))

            y_pred = classifier.predict_proba (np.array(X_fs_test, dtype = np.float32))
            assert(classifier.classes_ == [0,1]).all()
            y_pred = y_pred[:,1]

            # do not need testAugs here, as we mean it out anyway
            tmpY = pd.DataFrame([y_pred, testIDs], index=["pred", "Patient"]).T
            tmpY["pred"] = tmpY["pred"].astype(np.float32)
            y_pred = tmpY.groupby("Patient").mean()
            # reorder them to match y
            y_pred = y_pred.loc[baseX.iloc[test_index]["Patient"]]
            y_pred = y_pred["pred"].values

            # now we also need the true preds, y_fs_test is 'fat' with augmentations
            y_base_test = baseY.iloc[test_index]

            # and measure the error
            stats["fold_"+str(k)], df, acc = testModel (y_pred, y_base_test, idx = test_index, fold = k)
        else:
            # else we can just take 1 as a prediction
            y_pred = y_test*0 + 1
            stats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)
    stats = logMetrics (stats)

    # time
    timeFSEnd = time.time()
    stats["Time_Overall"] =  timeFSEnd - timeFSStart

    stats["eConfig"] = eConfig
    dump (stats, fResults)
    pass



def executeExperiments (z):
    fselExperiments, clfExperiments, data, dataID, expID, eConfig = z
    executeExperiment ([fselExperiments], [clfExperiments], data, dataID, expID, eConfig)


def getDataForExp (d, e, exception = True):
    csvName = os.path.join(featuresPath, d + "_" + e + ".csv")
    if os.path.exists(csvName) == False:
        if exception == True:
            raise Exception ("Not exists:" + str(csvName))
        return None
    data = pd.read_csv(csvName)
    # drop these now, other later
    data = data.drop(["Unnamed: 0", "Image", "mask"], axis = 1).copy()
    return (data)



if __name__ == "__main__":
    print ("Hi.")
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # generate experiments
    expList, expDict = getExperiments (deepParameters, radParameters)
    random.shuffle(expList)
    print ("Extracting", len(expList), "feature sets.")

    # generate all experiments
    fselExperiments = generateAllExperiments (fselParameters)
    print ("Created", len(fselExperiments), "feature selection parameter settings")
    clfExperiments = generateAllExperiments (clfParameters)
    print ("Created", len(clfExperiments), "classifier parameter settings")
    print ("Total", len(clfExperiments)*len(fselExperiments), "experiments")

    # load data first
    for dataID in dList:
        # search for feature sets
        for e, expID in enumerate(expDict):
            data = getDataForExp (dataID, expID, exception = False)
            if data is None:
                print ("No data for",dataID, expID)
                continue
            print ("Loaded data with shape", data.shape)
            # generate list of experiment combinations
            clList = []
            for fe in fselExperiments:
                for clf in clfExperiments:
                    eConfig = {"Fsel": fe, "Clf": clf}
                    clList.append( (fe, clf, data, dataID, expID, eConfig))
            print ("After ignoring:", len(clList), "experiments")
            # execute
            ncpus = 24
            fv = Parallel (n_jobs = ncpus)(delayed(executeExperiments)(c) for c in clList)



#
