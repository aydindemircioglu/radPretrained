#
import random
import pandas as pd
from radiomics import featureextractor

import json
import cv2
import nibabel as nib
import argparse
import logging
import math
import numpy as np
import itertools
from pathlib import Path

import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as tf
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models as models
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names


from helpers import *
from parameters import *
from joblib import Parallel, delayed


# global
nnModels = {}



class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('-f', type=str, default="dummy", help='dummy parameter')
        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt



def getNodeForPosition (position, modelType):
    if modelType == "ResNet50":
        if position == "Mid":
            node = "layer3.5.relu_2"
        elif position == "Top":
            node = "flatten"
        else:
            raise Exception ("Unknown feature extraction layer.")
    elif modelType == "ResNet18":
        if position == "Mid":
            node = "layer3.1.relu_1"
        elif position == "Top":
            node = "flatten"
        else:
            raise Exception ("Unknown feature extraction layer.")
    elif modelType == "DenseNet169":
        if position == "Mid":
            node = "features.transition3.pool"
        elif position == "Top":
            node = "flatten" # 22
        else:
            raise Exception ("Unknown feature extraction layer.")
    elif modelType == "VGG19":
        if position == "Mid":
            node = "features.26"
        elif position == "Top":
            node = "features.35" # 26
        else:
            raise Exception ("Unknown feature extraction layer.")
    else:
        raise Exception ("Unknown model!")
    return node



def getModel (config):
    architecture = config["Architecture"]
    if architecture in nnModels.keys():
        return nnModels[architecture]

    if architecture == "ResNet50":
        model = models.resnet50(pretrained = True).cuda()
        nnModels[architecture] = model
    elif architecture == "ResNet18":
        model = models.resnet18(pretrained = True).cuda()
        nnModels[architecture] = model
    elif architecture == "VGG19":
        model = models.vgg19(pretrained = True).cuda()
        nnModels[architecture] = model
    elif architecture == "DenseNet169":
        model = models.densenet169(pretrained = True).cuda()
        nnModels[architecture] = model
    else:
        raise Exception ("Unknown architecture")
    return model



def getFeatureExtractor (model, config):
    level = config["FeatureLevel"]
    nodeList = [getNodeForPosition ( level, config["Architecture"]) ]
    feature_extractor = create_feature_extractor(model, return_nodes= nodeList)
    return (feature_extractor)



def extractFeatureFromImage (fdict, config):
    nodeList = [getNodeForPosition ( config["FeatureLevel"], config["Architecture"]) ]
    fv = fdict[nodeList[0]]
    if len(fv.shape) == 4:
        #print(fv.size())
        fv = nn.AvgPool2d(fv.size()[-1])(fv)
    fv = fv.squeeze()
    return fv



def extractFeatures (dataset, dataID, config):
    dname = dict_hash(config)
    csvName = os.path.join(featuresPath, dataID + "_" + dname + ".csv")
    if os.path.exists(csvName) == True:
        print ("Exists. Touching it")
        Path(csvName).touch()
        return None

    # generic or not
    if config["Type"] == "Generic":
        # technically necessary to find correct resampled scan
        config["Resampling"] = 1
        pass
    elif config["Type"] == "Deep":
        m = getModel(config)
        fe = getFeatureExtractor(m, config).cuda()
    else:
        raise Exception ("Unknown config type for extraction.")


    data = dataset.copy()
    finalData = []

    for i, (idx, row) in enumerate(data.iterrows()):
        fvol = glob(os.path.join(cachePath, row["Patient"] +"*image*" + "_" + str(config["Resampling"]) + "*.nii.gz") )
        fmask = glob(os.path.join(cachePath, row["Patient"] +"*seg*" + "_" + str(config["Resampling"]) + "*.nii.gz") )
        print (fvol, fmask, row["Patient"])
        assert(len(fvol) == 1)
        assert(len(fmask) == 1)


        ### DEEP
        if config["Type"] == "Deep":
            # find largest slice

            vol = nib.load(fvol[0]).get_fdata()
            volMask = nib.load(fmask[0]).get_fdata()

            if volMask.shape != vol.shape:
                print (data, row)
                assert(volMask.shape == vol.shape)

            # find mask volume
            rmin, rmax, cmin, cmax, zmin, zmax = getBoundingBox(volMask)
            if rmax < rmin or cmax < cmin:
                print (rmin, rmax, cmin, cmax)
                print (row)
                raise Exception("Something is wrong with the bounding box")
            vol = vol[rmin:rmax, cmin:cmax, :]
            volMask = volMask[rmin:rmax, cmin:cmax, :]

            # convert intensities. -- global or local??
            if dataID in CT_datasets:
                vol[vol < -1024] = -1024
                vol[vol > 2048] = 2048
                vol = vol + 1024
                vol = vol/(2048+1024)
                # vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
                vol = np.asarray(255*vol, dtype = np.uint8)
            else:
                vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
                vol = np.asarray(255*vol, dtype = np.uint8)

            slices = []
            pVols = []
            for h in range(volMask.shape[2]):
                # append area
                a = np.sum(np.abs(volMask[:,:,h]))
                # less than 4 pixel does not make much sense, even for 5x5mm
                if a > 1:
                    pVols.append(a)
                if a > 2.5*2.5*np.pi:
                    slices.append( [h, a] )
            if len(slices) == 0:
                raise Exception ("Config" + str(config) + "has empty slices, data:" + str(row) + "pvols:" + str(pVols))
            slices.sort(key=lambda x: x[1])
            eSlices = None
            if config["Slices"] == "Max":
                eSlices = slices[-1:]
            elif config["Slices"] == "All":
                eSlices = slices
            else:
                raise Exception ("Unknown slices-to-select method")


            # create augmentations
            for r in replays[config["Augmentations"]]:
                feats = []
                for s in eSlices:
                    img = vol[:,:,s[0]]
                    mask = volMask[:,:,s[0]]
                    # mask needs to be resized to 255 for ROIchannel/ROIcut

                    mask = mask*255.0
                    aImg = img.copy()
                    assert (np.max(mask) < 256)
                    assert (np.min(mask) > -0.1)

                    if config ["ROIType"] == "ROI":
                        # roi without mask, its a grey, so convert to RGB
                        aImg = np.stack((aImg,)*3, axis=-1)
                    elif config ["ROIType"] == "ROIcut":
                        aImg = (mask*aImg).copy()
                        aImg = np.stack((aImg,)*3, axis=-1)
                        # roi with mask using as cut out
                    elif config ["ROIType"] == "ROIchannel":
                        aImg = np.stack((aImg,)*3, axis=-1)
                        aImg[:,:,1] = mask
                        aImg[:,:,2] = mask*0.5 + aImg[:,:,0]*0.5
                        # roi and mask as additional channel
                    else:
                        raise Exception ("Unknown roi type.")
                    aImg = A.ReplayCompose.replay(r, image=aImg)["image"]

                    f = extractFeatureFromImage (fe(aImg.cuda().unsqueeze(0)), config)
                    f = f.detach().cpu().numpy()
                    feats.append(f)
                    print ("X", end = '', flush = True)
                # aggregate features from extracted slices
                if config["Aggregation"] == "Max":
                    f = np.max(np.array(feats), axis = 0)
                elif config["Aggregation"] == "Mean":
                    f = np.mean(np.array(feats), axis = 0)
                else:
                    raise Exception ("Unknown Aggregation method.")

                # finally we have the feature vector
                f = {"feat_" + str(k):v for k,v in enumerate(f)}
                for k in row.keys():
                    f[k] = row[k]
                finalData.append(pd.DataFrame(f, index = [0]))

        ### GENERIC
        if config["Type"] == "Generic":
            featureList = {}
            if dataID in MR_datasets:
                params = os.path.join("config/MR.yaml")
            else:
                params = os.path.join("config/CT.yaml")

            try:
                label = 1
                eParams = config.copy()
                del eParams["Type"]
                del eParams["Resampling"]
                extractor = featureextractor.RadiomicsFeatureExtractor(params, **eParams)
                f = extractor.execute(fvol[0], fmask[0], label = label)

                # we prepare it all so it can be used directly, so remove all diagnostics here
                f = {p:f[p] for p in f if "diagnost" not in p}
                for k in row.keys():
                    f[k] = row[k]
                finalData.append(pd.DataFrame(f, index = [0]))
            except Exception as e:
                #f = pd.DataFrame([{"ERROR": patID}])
                print (fvol,fmask)
                print ("#### GOT AN ERROR!", e)
                raise Exception ("really wrong here?")


    finalData = pd.concat(finalData, axis = 0)
    finalData.to_csv(csvName)
    return finalData


def getNParameters ():
    modelA = models.resnet50(pretrained = True).cuda()
    modelB = models.resnet18(pretrained = True).cuda()
    modelC = models.vgg19(pretrained = True).cuda()
    modelD = models.densenet169(pretrained = True).cuda()
    print ("ResNet50", sum(p.numel() for p in modelA.parameters()))
    print ("ResNet18", sum(p.numel() for p in modelB.parameters()))
    print ("VGG19", sum(p.numel() for p in modelC.parameters()))
    print ("DenseNet169", sum(p.numel() for p in modelD.parameters()))
    pass



if __name__ == "__main__":
    print ("Hi.")

    # interprete command line options
    opt = BaseOptions().parse()

    # create paths
    os.makedirs( featuresPath, exist_ok = True)

    # ennumerate all configs
    expList, expDict = getExperiments (deepParameters, radParameters)

    print ("Extracting", len(expList), "feature sets.")
    random.shuffle(expList)
    ncpus = 6
    for dataID in dList:
        print ("### Processing", dataID)
        data = loadData (dataID, drop = False)
        fv = Parallel (n_jobs = ncpus)(delayed(extractFeatures)(data, dataID, config) for config in expList)

#
