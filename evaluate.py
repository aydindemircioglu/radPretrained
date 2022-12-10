#!/usr/bin/python3

from collections import OrderedDict
from datetime import datetime

from contextlib import contextmanager
from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

from glob import glob
from joblib import dump, load
from matplotlib import cm
from matplotlib import pyplot
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.transforms import Bbox
from PIL import Image
from PIL import ImageDraw, ImageFont
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Any
import copy
import cv2
import hashlib
import itertools
import json
import math
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import pylab
import random
import scipy.cluster.hierarchy as sch
import seaborn as sns
import shutil
import sys
import tempfile
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from scipy import stats


#from utils import *
from helpers import *
from parameters import *



# delong
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
pROC = importr('pROC')

def getCI (predsX):
    Y = predsX["y_true"].values
    scoresA = predsX["y_pred"].values
    lower, auc, upper = pROC.ci(Y, scoresA, direction = "<")
    return lower, auc, upper


def delongTest (predsX, predsY):
    Y = predsX["y_true"].values
    scoresA = predsX["y_pred"].values
    scoresB = predsY["y_pred"].values
    rocA = pROC.roc (Y, scoresA, direction = "<")
    rocB = pROC.roc (Y, scoresB, direction = "<")

    aucA = pROC.auc(Y, scoresA)
    aucB = pROC.auc(Y, scoresB)
    #print ("AUC A:" + str(aucA))
    #print ("AUC B:" + str(aucB))
    robjects.globalenv['rocA'] = rocA
    robjects.globalenv['rocB'] = rocB

    z = rpy2.robjects.packages.reval ("library(pROC);z = roc.test(rocA, rocB, method= 'delong', progress='none'); p = z$p.value")
    z = robjects.r.z
    p = robjects.r.p[0]
    return p, aucA, aucB



def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df



def mixedModelDeep (rTable):
    df = rTable.copy()
    df = rTable.loc[rTable.groupby(["Dataset", "Architecture", "FeatureLevel", "ROIType", "Slices", "Aggregation"])["AUC"].idxmax()]
    df = df.reset_index()
    vc = {'Dataset': '0 + C(Dataset)'}
    df.to_csv("./results/mixed_model.csv")
    md = smf.mixedlm("AUC ~ Architecture + FeatureLevel + ROIType + Slices + Aggregation", data=df, re_formula='1',vc_formula = vc, groups="Dataset")
    mdf = md.fit()
    mdf
    from statsmodels.stats.multitest import multipletests
    mdf.pvalues.values
    multipletests(mdf.pvalues, method="holm")[1]
    print(mdf.summary())
    tbl = results_summary_to_dataframe(mdf).round(3)
    tbl.to_excel("./paper/Table_5.xlsx")
    pass



def mixedModelGen (rTable):
    df = rTable.copy()
    df = rTable.loc[rTable.groupby(["Dataset", "Extraction"])["AUC"].idxmax()]
    df = df.reset_index()
    tmpf = [z.split("_") for z in list(df["Extraction"].values)]
    tmpf = list(zip(*tmpf))
    df["ExtractionType"] = tmpf[0]
    df["ExtractionN"] = tmpf[1]
    vc = {'Dataset': '0 + C(Dataset)'}
    # add some random noise to this. this is necessary because the best AUCs
    # do not differ for some combination and the mixed model runs into a
    # singularity problem. this noise does not change the results (rerun with
    # larger noise and/or different seed to see this)
    np.random.seed(42)
    df["AUC"] = df["AUC"] + np.random.normal(0.0, 0.0001, (len(df)))
    md = smf.mixedlm("AUC ~  C(ExtractionN, Treatment(reference='25'))*C(ExtractionType, Treatment(reference='binWidth'))", data=df, re_formula='1',vc_formula = vc, groups="Dataset")
    mdf = md.fit()
    print(mdf.summary())
    pass


# this is only used for Figure 2
def createFigure():
    recreatePath ("./results/pat/")
    pat = "Melanoma-032"
    slice = 132
    fvol = glob(os.path.join(cachePath, pat +"*image*" + "_1" + "*.nii.gz") )
    vol = nib.load(fvol[0]).get_fdata()

    for k in [slice-30, slice-15, slice, slice+15, slice+30]:
        s = vol[16:323-16,32:323-64, k].transpose()
        s = (s - np.min(s))/(np.max(s) - np.min(s))
        s = np.asarray(255*s, dtype = np.uint8)
        cv2.imwrite("./results/pat/" + str(k)+".jpg", s)



def addText (finalImage, text = '', org = (0,0), fontFace = '', fontSize = 12, color = (255,255,255)):
     # Convert the image to RGB (OpenCV uses BGR)
     #tmpImg = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
     tmpImg = finalImage
     pil_im = Image.fromarray(tmpImg)
     draw = ImageDraw.Draw(pil_im)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font, fill = color)
     #tmpImg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
     tmpImg = np.array(pil_im)
     return (tmpImg.copy())



def addBorder (img, pos, thickness):
    if pos == "H":
        img = np.hstack([255*np.ones(( img.shape[0],int(img.shape[1]*thickness), 3), dtype = np.uint8),img])
    if pos == "V":
        img = np.vstack([255*np.ones(( int(img.shape[0]*thickness), img.shape[1], 3), dtype = np.uint8),img])
    return img


# just read both figures and merge them
def join_corr_plots():
    fontFace = "Arial"

    imA = cv2.imread("./results/Figure_Corr_Deep.png")
    shapeA = imA.shape
    imA = addBorder (imA, "V", 0.075)
    imA = addText (imA, "(a)", (0,0), fontFace, 112, color= (0,0,0))

    imB = cv2.imread("./results/Figure_Corr_Generic.png")
    rRatio = imB.shape[1]/imB.shape[0]
    newShape = (int (rRatio*shapeA[0]), shapeA[0])
    imB = cv2.resize(imB, newShape)
    imB = addBorder (imB, "V", 0.075)
    imB = addText (imB, "(b)", (0,0), fontFace, 112, color=(0,0,0))

    imC = cv2.imread("./results/Figure_Corr_Deep_Generic.png")
    rRatio = imC.shape[1]/imC.shape[0]
    newShape = (int (rRatio*shapeA[0]), shapeA[0])
    imC = cv2.resize(imC, newShape)
    imC = addBorder (imC, "V", 0.075)
    imC = addText (imC, "(c)", (0,0), fontFace, 112, color=(0,0,0))

    imB = addBorder (imB, "H", 0.075)
    imC = addBorder (imC, "H", 0.175)
    imgU = np.hstack([imA, imB, imC])
    cv2.imwrite("./paper/Figure_3.png", imgU)



def getResults (dList, expType = "Deep"):
    expList, expDict = getExperiments (deepParameters, radParameters)
    cacheFile = "./results/results_" +  expType + ".feather"
    if os.path.exists(cacheFile) == False:
        rTable = []
        for dataID in dList:
            # load data to get # features
            for expID in expDict.keys():
                # only deep ones
                if expDict[expID]["Type"] != expType:
                    continue
                fResults = os.path.join(resultsPath, dataID + "_" + expID)
                fResults = os.path.join(fResults,  "*.dump")
                for f in glob (fResults):
                    stats = load(f)
                    timeOverall = stats["Time_Overall"]
                    nFeatures = stats["eConfig"]["Fsel"][0][1]["nFeatures"]
                    fsel = stats["eConfig"]["Fsel"][0][0]
                    fselParams = stats["eConfig"]["Fsel"][0][1]
                    del fselParams["nFeatures"]
                    clf = stats["eConfig"]["Clf"][0][0]
                    clfParams = stats["eConfig"]["Clf"][0][1]
                    eConfig = expDict[expID]
                    if expType == "Deep":
                        rTable.append({"Dataset": dataID, "expID": expID,
                            "dumpPath": f,
                            "N": stats["N"],
                            "Architecture": eConfig["Architecture"],
                            "FeatureLevel": eConfig["FeatureLevel"],
                            "ROIType": eConfig["ROIType"],
                            "Slices": eConfig["Slices"],
                            "Aggregation": eConfig["Aggregation"],
                            "Augmentations": eConfig["Augmentations"],
                            "FSel": fsel, "NFeatures": nFeatures,
                            "Clf": clf, "Clf_Parameters": str(clfParams),
                            "Accuracy": stats["Accuracy"], "AUC": stats["AUC"], "Specificity": stats["Spec"],
                            "Sensitivity": stats["Sens"],
                            "Time_Overall": timeOverall})
                    else:
                        if "binWidth" in eConfig.keys():
                            extraction = "binWidth_" + str(eConfig["binWidth"])
                        else:
                            extraction = "binCount_" + str(eConfig["binCount"])
                        rTable.append({"Dataset": dataID, "expID": expID,
                            "dumpPath": f,
                            "N": stats["N"],
                            "Extraction": extraction,
                            "FSel": fsel, "NFeatures": nFeatures,
                            "Clf": clf, "Clf_Parameters": str(clfParams),
                            "Accuracy": stats["Accuracy"], "AUC": stats["AUC"], "Specificity": stats["Spec"],
                            "Sensitivity": stats["Sens"],
                            "Time_Overall": timeOverall})

        rTable = pd.DataFrame(rTable).sort_values(["AUC"])
        pickle.dump (rTable, open(cacheFile,"wb"))
    else:
        print ("Restoring results")
        rTable = pickle.load(open(cacheFile, "rb"))
    return rTable



def generateAUCTable (deepResults, genResults):
    fTable = []
    worc = {"CRLM": 0.68, "Desmoid": 0.82, "GIST": 0.77, "HN":0.84, "Lipo": 0.83, "Liver": 0.81, "Melanoma": 0.51}
    worc95u = {"CRLM": 0.80, "Desmoid": 0.89, "GIST": 0.83, "HN":0.91, "Lipo": 0.91, "Liver": 0.87, "Melanoma": 0.62}
    worc95l = {k:np.round(worc[k] - (-worc[k] + worc95u[k]),2) for k in worc}
    ps = []
    dmodels = []
    gmodels = []
    for dataID in dList:
        bestDeepModel = deepResults.query("Dataset == @dataID").sort_values("AUC").iloc[-1]
        dstats = load(bestDeepModel["dumpPath"])

        bestGenModel = genResults.query("Dataset == @dataID").sort_values("AUC").iloc[-1]
        gstats = load(bestGenModel["dumpPath"])

        assert (gstats["preds"]["y_true"] == dstats["preds"]["y_true"]).all()
        pDG = np.round(delongTest (gstats["preds"], dstats["preds"])[0], 3)
        ps.append("(p = " + str(pDG) + ")")
        dmodels.append({"Architecture": bestDeepModel["Architecture"],
            "FeatureLevel": bestDeepModel ["FeatureLevel"],
            "Slices": bestDeepModel ["Slices"],
            "Aggregation": bestDeepModel ["Aggregation"],
            "ROI": bestDeepModel ["ROIType"]})
        gmodels.append(bestGenModel["Extraction"])
    dmodels = pd.DataFrame(dmodels)
    for k in dmodels.keys():
        fTable.append (dmodels[k].values)
    fTable.append ([np.round(deepResults.query("Dataset == @dataID")["AUC"].max(), 2) for dataID in dList])
    gmodels = [g.replace("_", ":") for g in gmodels]
    fTable.append (gmodels)
    fTable.append ([np.round(genResults.query("Dataset == @dataID")["AUC"].max(), 2) for dataID in dList])
    ds = [str(np.round(a - b,2)) for a,b in list(zip(*[fTable[len(fTable)-3], fTable[len(fTable)-1]]))]
    fTable.append(    [x+" " +y for x,y in zip(*[ds, ps])])

    fTable = pd.DataFrame(fTable).T
    # save as well
    fTable

    fTable.index = dList
    for d in worc.keys():
        fTable.at[d,5] = str(worc[d]) + " (" + str(worc95l[d])+"-"+str(worc95u[d])+ ")"
    fTable = fTable.reset_index(drop = False)
    fTable["X"] = [0]*fTable.shape[0]
    fTable.columns = ["Dataset"] + list(dmodels.keys()) + \
        ["AUC (deep features)", "Generic Model",
        "AUC (generic features)",
        "P (deep vs generic)",
        "AUC (results from CITE)"]
    fTable.to_excel("./paper/Table_4.xlsx")
    fTable
    return fTable


def penaltyHits (deepResults):
    z = deepParameters.copy()
    del z["Resampling"]
    del z["Augmentations"]
    for p in z.keys():
        print("\n\n###", p)
        loss = {f: 1.0 for f in z[p]}
        for d in dList:
            bestAUC = deepResults.query("Dataset == @d").sort_values(["AUC"]).iloc[-1]["AUC"]
            for f in z[p]:
                subTbl = eval('deepResults.query("' + p + '!= @f' + '")')
                tmpAUC = subTbl.query("Dataset == @d").sort_values(["AUC"]).iloc[-1]["AUC"]
                loss[f] = np.min([loss[f], tmpAUC - bestAUC])
        for f in loss.keys():
            print (f, np.round(loss[f],3))
        #print (d, f, np.round(tmpAUC - bestAUC, 3))

    # remove what we do not want
    dR = deepResults.query ("Architecture != 'VGG19'").copy()
    dR = dR.query ("Architecture != 'ResNet18'")
    dR = dR.query ("Slices == 'All'")

    # show performances
    for d in dList:
        bestAUC = deepResults.query("Dataset == @d").sort_values(["AUC"]).iloc[-1]["AUC"]
        ebestAUC = dR.query("Dataset == @d").sort_values(["AUC"]).iloc[-1]["AUC"]
        print(d, bestAUC, ebestAUC)

    # remove what we do not want
    dR = genResults.copy()
    dR = dR.query ("Extraction != 'binWidth_100'")
    dR = dR.query ("Extraction != 'binWidth_10'")
    dR = dR.query ("Extraction != 'binCount_10'")
    dR = dR.query ("Extraction != 'binCount_100'")
    # show performances
    for d in dList:
        bestAUC = genResults.query("Dataset == @d").sort_values(["AUC"]).iloc[-1]["AUC"]
        ebestAUC = dR.query("Dataset == @d").sort_values(["AUC"]).iloc[-1]["AUC"]
        print(d, bestAUC, ebestAUC)


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


def getCs (setA, setB):
    setASets = setA["expID"].unique()
    setBSets = setB["expID"].unique()

    print ("Correlation matrix size", len(setASets), "x", len(setBSets))
    Cs = []
    for dataID in dList:
        C = np.zeros((len(setASets), len(setBSets)))
        for i, dA in enumerate(setASets):
            fsetA = getDataForExp (dataID, dA)
            fsetA = fsetA.drop(["Patient", "Target"], axis = 1)
            for j, dB in enumerate(setBSets):
                fsetB = getDataForExp (dataID, dB)
                fsetB = fsetB.drop(["Patient", "Target"], axis = 1)
                F = pd.concat([fsetA, fsetB], axis = 1).T

                CAB = np.corrcoef (F)[0:fsetA.shape[1], fsetA.shape[1]:fsetA.shape[1]+fsetB.shape[1]]
                C[i,j] = (np.nanmean(np.nanmax(CAB, axis = 1)) + np.nanmean(np.nanmax(CAB, axis = 0)))/2.0
        Cs.append(C)
    return Cs, setASets, setBSets



def plotCorrMatrix (Cs, stype):
    C = np.nanmean(Cs, axis = 0)
    f, ax = plt.subplots(figsize=(11, 9))
    #cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = sns.color_palette("vlag", as_cmap=True)
    cbar = True
    if stype != "Deep_Generic":
        cbar=False
    sns.heatmap(C, cmap=cmap, center=0.5, yticklabels=False,  xticklabels=False, vmin = 0, vmax = 1,
            square=True, linewidths=.5, cbar_kws={"shrink": .9, "pad": 0.2}, cbar = cbar)

    if stype == "Deep_Generic":
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)

    f.savefig("./results/Figure_Corr_" + str(stype) + ".png", dpi = DPI, facecolor = 'w', bbox_inches='tight')
    plt.close('all')
    pass



def statsCorr (C, sname = ''):
    C = np.nanmean(C, axis = 0)
    if C.shape[0] == C.shape[1]:
        for i in range(C.shape[0]):
            C[i,i] = np.nan

    print (sname, ":", np.round(np.nanmean(C),2), ";", np.round(np.nanmin(C),2), "-", np.round(np.nanmax(C),2))
    pass


def computeCorrelations(deepResults, genResults):
    cacheFile = "./results/rcorr_deep.feather"
    if os.path.exists(cacheFile) == False:
        Cs = getCs (deepResults, deepResults)
        pickle.dump (Cs, open(cacheFile,"wb"))
    else:
        print ("Restoring results")
        Cs = pickle.load(open(cacheFile, "rb"))
    statsCorr (Cs[0], "Deep-Deep")
    plotCorrMatrix (Cs[0], "Deep")

    # get rad correlations
    cacheFile = "./results/rcorr_generic.feather"
    if os.path.exists(cacheFile) == False:
        Cs = getCs (genResults, genResults)
        pickle.dump (Cs, open(cacheFile,"wb"))
    else:
        print ("Restoring results")
        Cs = pickle.load(open(cacheFile, "rb"))
    statsCorr (Cs[0], "Gen-Gen")
    plotCorrMatrix (Cs[0], "Generic")

    # get cross thing
    cacheFile = "./results/rcorr_cross.feather"
    if os.path.exists(cacheFile) == False:
        Cs = getCs (deepResults, genResults)
        pickle.dump (Cs, open(cacheFile,"wb"))
    else:
        print ("Restoring results")
        Cs = pickle.load(open(cacheFile, "rb"))
    statsCorr (Cs[0], "Deep-Gen")
    plotCorrMatrix (Cs[0], "Deep_Generic")
    return Cs


def test_getBestForCombo(deepResults):
    for dataID in dList:
        dR = deepResults.query("Dataset == @dataID and Architecture == 'ResNet50' and FeatureLevel == 'Top' and ROIType == 'ROIchannel' and Slices == 'All' and Aggregation == 'Mean' ")
        dR = dR.sort_values(["AUC"])
        print (dataID, dR.iloc[-1]["AUC"])


def getBestTable (rTable):
    bestTable = []
    for dataID in dList:
        bestTable.append(rTable.query("Dataset == @dataID").sort_values(["AUC"]).iloc[-1:])
    bestTable = pd.concat([pd.DataFrame(k) for k in bestTable])

    bestTable = bestTable.drop(["expID", "dumpPath"],axis=1).reset_index()
    bestTable = bestTable[["Dataset", "Architecture", "FeatureLevel", "ROIType", "Slices", "Aggregation", "AUC"]]
    bestTable.to_csv("./results/bestTable.csv")
    print(bestTable)
    return bestTable


def bin_total(y_true, y_prob, n_bins):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

    # In sklearn.calibration.calibration_curve,
    # the last value in the array is always 0.
    binids = np.digitize(y_prob, bins) - 1

    return np.bincount(binids, minlength=len(bins))



def getCalibrationPlots (rTable):
    strategy = "quantile"
    n_bins = 4
    fig, ax = plt.subplots(4,3, figsize = (12, 16), dpi = DPI)
    for k, dataID in enumerate(dList):
        j = k % 3
        i = k // 3
        N = bestDeepModel["N"].values[0]
        bestDeepModel = rTable.query("Dataset == @dataID").sort_values(["AUC"]).iloc[-1:]
        dstats = load(bestDeepModel["dumpPath"].values[0])
        x = dstats["preds"]["y_true"]
        y = dstats["preds"]["y_pred"]
        cc_x, cc_y = calibration_curve(x, y, n_bins = n_bins, strategy = strategy)
        #print(bin_total(x, y, n_bins=5))

        # only these two lines are calibration curves
        ax[i][j].plot(cc_x,cc_y, marker='o', linewidth=1, label='')
        #plt.plot(logreg_x,logreg_y, marker='o', linewidth=1, label='')

        # reference line, legends, and axis labels
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax[i][j].transAxes
        line.set_transform(transform)
        ax[i][j].set_xlim(0,1)
        ax[i][j].set_ylim(0,1)
        ax[i][j].add_line(line)
        ax[i][j].set_title(dataID)
        ax[i][j].set_xlabel('Predicted probability')
        ax[i][j].set_ylabel('Fraction of positives')
    fig.delaxes(ax[3][2])
    fig.delaxes(ax[3][1])
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    #plt.tight_layout()
    fig.savefig("./results/Figure_Calibration.png", facecolor = 'w', bbox_inches='tight')
    pass


def getRecommendationTable (rTable):
    # recommendation lost?
    recTable = []
    for dataID in dList:
        bOne = rTable.query("Dataset == @dataID")
        bOne = bOne.query("Architecture == 'ResNet50' ")
        bOne = bOne.query("FeatureLevel == 'Mid' ")
        bOne = bOne.query("ROIType == 'ROIchannel' ")
        bOne = bOne.query("Slices == 'All' ")
        bOne = bOne.query("Aggregation == 'Mean' ")
        recTable.append(bOne.sort_values(["AUC"]).iloc[-1:])
    recTable = pd.concat([pd.DataFrame(k) for k in recTable])
    recTable = recTable[["Dataset", "Architecture", "FeatureLevel", "ROIType", "Slices", "Aggregation", "AUC"]]
    bestTable["Rec-AUC"] = recTable["AUC"].values
    print(bestTable)
    pass


def generateBestOfPlots (deepResults, ptype = "strip"):
    z = deepParameters.copy()
    del z["Resampling"]
    del z["Augmentations"]

    dfmax = deepResults.copy()
    dfmax = dfmax.loc[dfmax.groupby(["Dataset", "Architecture", "FeatureLevel", "ROIType", "Slices", "Aggregation"])["AUC"].idxmax()]
    dfmax = dfmax.reset_index()
    for p in z.keys():
        pdata = []
        for d in dList:
            for k in z[p]:
                df = dfmax.query("Dataset == @d").copy()
                subTbl = eval('df.query("' + p + '== @k' + '")')
                pdata.append(subTbl)

        pdata = pd.concat(pdata)
        fig, ax = plt.subplots(figsize = (12, 10), dpi = DPI)

        if ptype == "bar":
            cp = sns.color_palette("Spectral", 4)
            sns.barplot(hue=p,
                        y="AUC",
                        ci = "sd",
                        x="Dataset",
                        palette = cp,
                        data=pdata)
        if ptype == "strip":
            cp = sns.color_palette("hls", 4)
            sns.stripplot(hue=p,
                        y="AUC",
                        x="Dataset",
                        jitter = 0.2,
                        palette = cp,
                        data=pdata)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('AUC', fontsize = 22, labelpad = 12)
        plt.xlabel('Dataset', fontsize= 22, labelpad = 12)
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title
        #ax.set_xticks(nList[1:])#, rotation = 0, ha = "right", fontsize = 22)

        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        plt.tight_layout()
        fig.savefig("./results/Figure_"+p+".png", facecolor = 'w', bbox_inches='tight')


if __name__ == "__main__":
    print ("Hi.")

    # obtain results
    print ("Generating results")

    deepResults = getResults (dList, "Deep")
    genResults = getResults (dList, "Generic")

    mixedModelDeep (deepResults)

    # correlations
    computeCorrelations (deepResults, genResults)
    join_corr_plots()

    # table
    generateAUCTable (deepResults, genResults)

    # plots
    generateBestOfPlots (deepResults)

    # simple stats
    getBestTable (deepResults)

    # calibration plots
    getCalibrationPlots (deepResults)

#
