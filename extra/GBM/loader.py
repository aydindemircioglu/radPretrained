from glob import glob
import numpy as np
import os
import pandas as pd
import shutil


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


if __name__ == '__main__':
    # https://github.com/JSB-UCLA/ITCA/blob/378510815de7693878264bcce5184eb9ce4bc8ad/data/application2/gbm_tcga.csv
    tbl = pd.read_csv("./gbm_tcga.csv")
    tbl["ID"] = tbl["Patient ID"]

    mainPath = "./GBM"
    recreatePath (mainPath)

    infoTbl = []
    for i, (idx, row) in enumerate(tbl.iterrows()):
        diag = None
        if row["MGMT Status"] == "UNMETHYLATED":
            diag = 0
        if row["MGMT Status"] == "METHYLATED":
            diag = 1
        if diag is None:
            continue
        try:
            patID = row["Patient ID"]
            img = glob(os.path.join(patID, "*_t1.nii*"))[0]
            mask = None
            try:
                mask = glob(os.path.join(patID, "*Manually*"))[0]
            except:
                mask = glob(os.path.join(patID, "*GlistrBoost*"))[0]
            print (img, mask)
            patID = patID.replace("TCGA", "GBM")
            tPath = os.path.join("GBM", patID)
            os.makedirs (tPath)
            shutil.copyfile (img, os.path.join(tPath, "Image.nii.gz"))
            shutil.copyfile (img, os.path.join(tPath, "Mask.nii.gz"))
        except Exception as e:
            #raise Exception (e)
            continue
            pass
        infoTbl.append({"Patient": patID, "Diagnosis": diag})

    infoTbl = pd.DataFrame(infoTbl)
    infoTbl.to_csv("./pinfo_GBM.csv", index = False)


#
