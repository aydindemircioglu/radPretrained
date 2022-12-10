from collections import OrderedDict
import numpy as np
import os
import cv2
import itertools
import albumentations as A
import torch
import random
from albumentations.pytorch.transforms import ToTensorV2
from joblib import parallel_backend, Parallel, delayed, load, dump


### parameters
DPI = 300
oneDocument = False



nRepeats = 1
nCV = 10


basePath = "/home/aydin/results/radPretrained"
ncpus = 16

dList = [ "Desmoid", "Lipo", "Liver", "Melanoma", "GIST", "CRLM", "HN", "GBM", "C4KCKiTS", "ISPY1" ]
dList = sorted(dList)

CT_datasets = ['HN', 'GIST', 'CRLM', 'Melanoma', 'C4KCKiTS']
MR_datasets = ['Lipo', 'Desmoid', 'Liver', 'ISPY1', 'GBM']

#dList = ["Lipo"]

imagePath = "/data/radDatabase"
featuresPath = os.path.join(basePath, "features")
cachePath = os.path.join(basePath, "cache")
resultsPath = os.path.join(basePath, "results")



# just for validation
valTransforms = A.Compose([
    A.Normalize(always_apply=True),
    ToTensorV2()
])

# no augmentation
noTransforms = A.ReplayCompose([
    A.Resize(224, 224, always_apply=True),
    A.Normalize(always_apply=True),
    ToTensorV2()
])


# create replays for augmentations, but we dont use them here.
if os.path.exists("./results/augmentations.replay") == False:
    ## create no augmentation
    replays = {}
    replays["None"] = [noTransforms(image = img)["replay"]]
    dump (replays, "./results/augmentations.replay")
else:
    replays = load ("./results/augmentations.replay")



radParameters = OrderedDict({
    # these are 'one-of'
    "binWidth": [10,25,50,100],
    "binCount": [10,25,50,100],
})



deepParameters = OrderedDict({
    # these are 'one-of'
    "Architecture": ["ResNet18", "ResNet50", "VGG19", "DenseNet169"],
    "FeatureLevel": ["Top", "Mid"],
    "Resampling": [1], #1, 3, 5], 5mm makes no sense, and why 3mm, we dont have PET anyway
    "ROIType": ["ROI", "ROIcut", "ROIchannel"],
    "Slices": ["Max", "All"],
    "Aggregation": ["Max", "Mean"],
    "Augmentations": ["None"] # wanted to do this, but it will take forever on my hardware. bad luck.
})



fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32,64],
        "Methods": {
            "ET": {},
            "LASSO": {"C": [1.0]},
            "Anova": {},
            "tScore": {},
            "Bhattacharyya": {},
            "RF": {}
        }
    }
})



clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "RBFSVM": {"C":np.logspace(-6, 6, 7, base = 2.0), "gamma":["auto"]},
            "RandomForest": {"n_estimators": [50, 125, 250]},
            "LogisticRegression": {"C": np.logspace(-6, 6, 7, base = 2.0) },
            "NeuralNetwork": {"layer_1": [4, 16, 64], "layer_2": [4, 16, 64], "layer_3": [4, 16, 64]},
            "NaiveBayes": {}
        }
    }
})


#
