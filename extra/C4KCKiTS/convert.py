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



def getSeries(row):
    global segExec
    try:
        nifti_path = os.path.join(mainPath, row["ID"])
        imgPath = os.path.join(mainPath, os.path.basename(row["ID"]), "Image.nii.gz")
        maskPath = os.path.join(mainPath, os.path.basename(row["ID"]), "Mask.nii.gz")
        if not os.path.exists(nifti_path):
            os.makedirs(nifti_path)

        dicom2nifti.dicom_series_to_nifti(row["Series"], imgPath, reorient_nifti=False)
        segDICOM = glob(row["Segmentation"]+"/*")[0]
        execStr = "segimage2itkimage --inputDICOM " + segDICOM + " -t nii"
        execStr += ' --outputDirectory ' + nifti_path
        segExec.append(execStr)
    except Exception as e:
        print ("[red]#### uncaught ERROR: ", e, "[/red]")
        raise(e)
        return None



if __name__ == '__main__':
    # read clinical data
    tbl = pd.read_csv("./C4KCKiTS.csv")
    mainPath = "./C4KCKiTS"
    recreatePath (mainPath)
    infoTbl = []
    segExec = []
    for i, (idx, row) in enumerate(tbl.iterrows()):
        try:
            newRow = getSeries(row)
            pass
        except Exception as e:
            raise Exception (e)
        infoTbl.append({"Patient": row["ID"], "Diagnosis": row["Subtype"]})
    infoTbl = pd.DataFrame(infoTbl)
    infoTbl.to_csv("./pinfo_C4KCKiTS.csv", index = False)
    pd.DataFrame(segExec).to_csv("./exec_seg.sh", index = False, header = None)

#
