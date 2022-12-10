#
import pandas as pd
from rich import print
from termcolor import colored
from joblib import Parallel, delayed
import os
import multiprocessing
import time
import multiprocessing.pool
import functools
from glob import glob
import pydicom
import SimpleITK as sitk
import nibabel as nib
import nibabel.processing as nibp
from progressbar import progressbar

from parameters import *
from helpers import *



def getFout (f):
    if "HN-" in f:
        f = f.replace("mask_", "seg_")
        pdir = f.split("/")[6].split("_")[0]
        fout = os.path.join(pdir + "_" + os.path.basename(f).replace(".nii.gz", "_" + str(p) + ".nii.gz"))
        fout = os.path.join(cachePath,  fout)
    else:
        pdir = (os.path.basename(os.path.dirname(f)))
        fout = os.path.join(pdir + "_" + os.path.basename(f).replace(".nii.gz", "_" + str(p) + ".nii.gz"))
        fout = os.path.join(cachePath,  fout)

    # fix for KiTS
    if "KiTS-" in f:
        fout = fout.replace("_2_1.", "_segmentation_1.")
        fout = fout.replace("Image_", "image_")
    # stupid
    if "GBM-" in f or "ISPY-" in f:
        fout = fout.replace("_Mask_", "_segmentation_")
        fout = fout.replace("_Image_", "_image_")

    return fout



def processFile (row, p):
    f = row["Image"]
    fmask = row["mask"]

    fout_img = getFout (f)
    fout_mask = getFout (fmask)

    if os.path.exists(fout_img):
        return None

    img = nib.load(f)
    seg = nib.load(fmask)

    # make sure the mask is all 0 and 1
    tmp = seg.get_fdata()
    tmp = np.asarray(tmp > 0, dtype = np.uint8)

    # copy also over affine infos, else we can end up with volumes that are SLIGHTLY different
    new_seg = seg.__class__(tmp, img.affine, img.header)
    rimg = nibp.resample_to_output(img, voxel_sizes = [p, p, p], order = 3)
    rSeg = nibp.resample_to_output(new_seg, voxel_sizes = [p, p, p], order = 0)

    test = np.median(rSeg.get_fdata())
    assert (test == 0.0 or test == 1.0)

    #print (f, rimg.shape, rSeg.shape)
    assert (rimg.shape == rSeg.shape)
    print ("X", end = '', flush = True)

    nib.save(rimg, fout_img)
    nib.save(rSeg, fout_mask)
    pass



if __name__ == '__main__':
    #recreatePath (cachePath)

    for d in dList:
        data = loadData (d, drop = True)
        for p in deepParameters ["Resampling"]:
            fv = Parallel (n_jobs = ncpus)(delayed(processFile)(row, p) for (idx, row) in data.iterrows())

#
