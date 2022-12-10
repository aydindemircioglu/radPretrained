from glob import glob
import numpy as np
import os
import pandas as pd

src = "./data.csv"

data = pd.read_csv("./data.csv")
data["ID"] = data["case_id"]
stats = {"subtype": [], "stage": []}


# detect all arterial images and segmentations
pats = glob("manifest-1592488683281/C4KC-KiTS/*")
tbl = []
for p in pats:
    ID = os.path.basename(p)
    series = glob (p+"/*/*")
    arterial = None
    seg = None
    for s in series:
        if "arteria" in s:
            arterial = s
        if "Segment" in s:
            seg = s
    if arterial is not None and seg is not None:
        pID = ID.replace("KiTS-", "case")
        rows = data.query("ID == @pID")
        if len (rows) >0 :
            for k in stats:
                exec(k + "=" + "rows.iloc[0]['" + k + "']")
                stats[k].append(rows.iloc[0][k])
        else:
            print (ID, len(rows))
            pass

        tbl.append({"Series": arterial, "Segmentation": seg, "Subtype": subtype, "Stage": stage, "ID":ID} )

for k in stats:
    print (k, len(stats[k]), sum([x is np.nan for x in stats[k]]))

# we only want stage or subtype?
# no idea, we take subtype.
#tbl = tbl.query("Stage > -1")
tbl = pd.DataFrame(tbl)
tbl = tbl.query("Subtype > -1")

tbl.to_csv ("./C4KCKiTS.csv")

#
