from sklearn.metrics import roc_curve, auc, roc_auc_score
from math import sqrt
import numpy as np
import scipy.stats
from scipy import stats
import shutil
import os
from glob import glob
from pprint import pprint
from skimage.measure import label
import pandas as pd
from parameters import *
from typing import Dict, Any
import cProfile
import pstats
from functools import wraps
import numpy as np
import random
import json
import hashlib


def dict_hash(dictionary: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def getExperiments (deepParameters, radParameters):
    # create deep sets
    expList = [dict(zip(deepParameters.keys(), z) ) for z in list(itertools.product(*deepParameters.values()))]
    for v in range(len(expList)):
        expList[v]["Type"] = "Deep"

    # remove those that have slices=max and aggregation=mean
    print ("Before", len(expList))
    newExpList = []
    for v in range(len(expList)):
        expList[v]["Type"] = "Deep"
        if expList[v]["Aggregation"] == "Max" and expList[v]["Slices"] == "Max":
            continue
        newExpList.append(expList[v])
    expList = newExpList
    print ("After", len(expList))

    # create generic sets
    rexpList = [{k:v} for k in radParameters.keys() for v in radParameters[k]]
    for v in range(len(rexpList)):
        rexpList[v]["Type"] = "Generic"

    expList.extend(rexpList)

    expDict = {}
    for e in expList:
        dname = dict_hash(e)
        expDict[dname] = e

    return expList, expDict



# blacklist comes from another project,
# contains either those not processable by pyradiomics
# or those with too large slice thickness
def loadData (dataID, drop = True):
    # load data first
    data = pd.read_csv("./data/pinfo_" + dataID + ".csv")
    blacklist = pd.read_csv("./data/blacklist.csv").T.values[0]
    data = data.query("Patient not in @blacklist").copy()

    # fix this
    if dataID == "HN":
        data["Diagnosis"] = data["Tstage"]
        data = data.drop (["Tstage"], axis = 1).copy()

    # add path to data
    for i, (idx, row) in enumerate(data.iterrows()):
        image, mask = getImageAndMask (dataID, row["Patient"])
        data.at[idx, "Image"] = image
        data.at[idx, "mask"] = mask
    print ("### Data shape", data.shape)

    data["Target"] = data["Diagnosis"]
    data = data.drop(["Diagnosis"], axis = 1).reset_index(drop = True).copy()
    if drop == True:
        data = data.drop(["Patient"], axis = 1).reset_index(drop = True).copy()

    # make sure we shuffle it and shuffle it the same
    np.random.seed(111)
    random.seed(111)
    data = data.sample(frac=1)

    return data



def getImageAndMask (d, patName):
    image = os.path.join(imagePath, patName, "image.nii.gz")
    if d in ["HN"]:
        # nifti only exists with CT, so PET will be ignored
        cands = glob(os.path.join(imagePath, patName + "*/**/image.nii.gz"), recursive = True)
        if len(cands) != 1:
            print ("Error with ", patName)
            print ("Checked", os.path.join(imagePath, patName + "*/**/image.nii.gz"))
            pprint(cands)
            raise Exception ("Cannot find image.")
        image = cands[0]
        cands = glob(os.path.join(imagePath, patName + "*/**/mask_GTV-1.nii.gz"), recursive = True)
        if len(cands) != 1:
            print ("Error with ", patName)
            pprint(cands)
            raise Exception ("Cannot find mask.")
        mask = cands[0]
    if d in ["Desmoid", "GIST", "Lipo", "Liver"]:
        mask = os.path.join(imagePath, patName, "segmentation.nii.gz")
    if d in ["CRLM"]:
        mask = os.path.join(imagePath, patName, "segmentation_lesion0_RAD.nii.gz")
    if d in ["Melanoma"]:
        mask = os.path.join(imagePath, patName, "segmentation_lesion0.nii.gz")


    if d in ["ISPY1", "GBM"]:
        # 2 is the renal tumor
        image = os.path.join(imagePath, patName, "Image.nii.gz")
        mask = os.path.join(imagePath, patName, "Mask.nii.gz")

    if d in ["C4KCKiTS"]:
        # 2 is the renal tumor
        image = os.path.join(imagePath, patName, "Image.nii.gz")
        mask = os.path.join(imagePath, patName, "2.nii.gz")


    # special cases
    if patName == "GIST-018":
        image = os.path.join(imagePath, patName, "image_lesion_0.nii.gz")
        mask = os.path.join(imagePath, patName, "segmentation_lesion_0.nii.gz")
    if patName == "Lipo-073":
        mask = os.path.join(imagePath, patName, "segmentation_Lipoma.nii.gz")

    if os.path.exists(image) == False:
        print ("Missing", image)
    if os.path.exists(mask) == False:
        print ("Missing", mask)
    return image, mask


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC = 255*largestCC
    return np.asarray(largestCC, dtype = np.uint8)

# find bounding box around mask
def getMaskExtend (sliceMask):
    def fix (v, b):
        v = 0 if v < 0 else v
        v = b-1 if v > b-1 else v
        return v
    x0 = np.min(np.where(np.sum(sliceMask,axis=0))) - 16
    x1 = np.max(np.where(np.sum(sliceMask,axis=0))) + 16
    y0 = np.min(np.where(np.sum(sliceMask,axis=1))) - 16
    y1 = np.max(np.where(np.sum(sliceMask,axis=1))) + 16
    x0, x1, y0, y1 = fix(x0, sliceMask.shape[1]), fix(x1, sliceMask.shape[1]), fix(y0, sliceMask.shape[0]), fix(y1, sliceMask.shape[0])
    return x0, x1, y0, y1

def extractMaskAndImage (sliceMask, slice):
    x0, x1, y0, y1 = getMaskExtend (sliceMask)
    return sliceMask[y0:y1, x0:x1], slice[y0:y1, x0:x1]



def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity




# https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
  """Return an axes of confidence bands using a simple approach.

  Notes
  -----
  .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
  .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

  References
  ----------
  .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
     http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

  """
  if ax is None:
      ax = plt.gca()

  ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
  ax.fill_between(x2, y2 + ci, y2 - ci, color="#111111", edgecolor=['none'], alpha =0.15)

  return ax


# Modeling with Numpy
def equation(a, b):
  """Return a 1D polynomial."""
  return np.polyval(a, b)



def full_extent(ax, pad=0.0):
  """Get the full extent of an axes, including axes labels, tick labels, and
  titles."""
  # For text objects, we need to draw the figure first, otherwise the extents
  # are undefined.
  ax.figure.canvas.draw()
  items = ax.get_xticklabels() + ax.get_yticklabels()
  items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
  items += [ax, ax.title]
  bbox = Bbox.union([item.get_window_extent() for item in items])
  return bbox.expanded(1.0 + pad, 1.0 + pad)



def recreatePath (path, create = True):
        print ("Recreating path ", path)
        try:
                shutil.rmtree (path)
        except:
                pass

        if create == True:
            try:
                    os.makedirs (path)
            except:
                    pass
        print ("Done.")



# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def getBoundingBox(img, expFactor = 0.1):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    rmin = np.max([int(rmin - (rmax-rmin)*expFactor) , 0])
    cmin = np.max([int(cmin - (cmax-cmin)*expFactor), 0])
    rmax = np.min([int(rmax + (rmax - rmin)*expFactor), img.shape[0]])
    cmax = np.min([int(cmax + (cmax-cmin)*expFactor), img.shape[1]])

    return rmin, rmax, cmin, cmax, zmin, zmax

#
