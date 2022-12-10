import pandas as pd
#from rich import print
#from termcolor import colored
import os
from DicomRTTool import DicomReaderWriter
import multiprocessing
import shutil
import time
import multiprocessing.pool
import functools
from glob import glob
import pydicom
import SimpleITK as sitk
import dicom2nifti
import nibabel as nib


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



# def getSeries(row, imgPath, segPath):
#     try:
#         img = glob(os.path.join(imgPath, "ISPY1_" + str(row["SUBJECTID"])) + "/*DCE_0001*")[0]
#         mask = os.path.join(segPath, "ISPY1_" + str(row["SUBJECTID"]) + ".nii.gz")
#
#         # # everything is sooo f..ked up
#         # i = nib.load(img)
#         # rimg = nibp.resample_to_output(i, voxel_sizes = [1, 1, 1], order = 3)
#         # m = nib.load(mask)
#         # # tmp = m.get_fdata()
#         # # tmp = np.asarray(tmp > 0, dtype = np.uint8)
#         # rimgN = rimg.__class__(rimg.get_fdata(), m.affine, m.header)
#         # print (rimgN.shape)
#         # print (m.shape)
#         # # target
#         tPath = os.path.join(mainPath, os.path.basename("ISPY1-"+str(row["SUBJECTID"])))
#         if not os.path.exists(tPath):
#             os.makedirs(tPath)
#
#         imgPath = os.path.join(mainPath, os.path.basename("ISPY1-"+str(row["SUBJECTID"])), "Image.nii.gz")
#         maskPath = os.path.join(mainPath, os.path.basename("ISPY1-"+str(row["SUBJECTID"])), "Mask.nii.gz")
#         nib.save(rimgN, imgPath)
#         nib.save(m, maskPath)
#
#         return "ISPY1-"+str(row["SUBJECTID"]), imgPath, maskPath
#     except Exception as e:
#         print(e)
#         raise(e)
#         return None, None, None



def getSeries(row, imgPath, segPath):
    try:
        img = glob(os.path.join(imgPath, "ISPY1_" + str(row["SUBJECTID"])) + "/*DCE_0001*")[0]
        mask = os.path.join(segPath, "ISPY1_" + str(row["SUBJECTID"]) + ".nii.gz")

        # target
        tPath = os.path.join(mainPath, os.path.basename("ISPY1-"+str(row["SUBJECTID"])))
        if not os.path.exists(tPath):
            os.makedirs(tPath)

        imgPath = os.path.join(mainPath, os.path.basename("ISPY1-"+str(row["SUBJECTID"])), "Image.nii.gz")
        maskPath = os.path.join(mainPath, os.path.basename("ISPY1-"+str(row["SUBJECTID"])), "Mask.nii.gz")
        shutil.copyfile (img, imgPath)
        shutil.copyfile (mask, maskPath)
        return "ISPY1-"+str(row["SUBJECTID"]), imgPath, maskPath
    except Exception as e:
        return None, None, None



if __name__ == '__main__':
    # read clinical data
    tbl = pd.read_excel("./I-SPY 1 All Patient Clinical and Outcome Data.xlsx", sheet_name = 1)

    imgPath = "NIfTI-Files/images_bias-corrected_resampled_zscored_nifti/"
    segPath = "NIfTI-Files/masks_stv_manual/"

    mainPath = "./ISPY1"
    recreatePath (mainPath)

    infoTbl = []
    for i, (idx, row) in enumerate(tbl.iterrows()):
        try:
            ID, iP, mP = getSeries(row, imgPath, segPath)
            if ID is None:
                continue
            infoTbl.append({"Patient": ID, "Diagnosis": int(row["HR Pos"])})
        except:
            # possible HR Pos = NA
            continue
    infoTbl = pd.DataFrame(infoTbl)
    infoTbl.to_csv("./pinfo_ISPY1.csv", index = False)


# for g in glob("./ISPY1/*/"):
#     n = nib.load(g+"/Image.nii.gz")
#     print (n.header.get_zooms()[2])
#


#
