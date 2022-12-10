first use the loader.py it will create a C4KCKiTS.csv
then use convert.py to convert them and write them to ./C4KCKiTS/
segmentation are not converted in python, but convert.py creates a exec_seg.sh
just start it, but make sure, dcmqi is available in /usr/local/bin

there are two segmentations in each DCM-SEG, the first one is
the whole kidney, the second one is the renal tumor.
thus, when loading the data, we need to extract 2.nii.gz from each series.

but well, after convert we want to resample the data,
we do this in the project

KiTS-00095/, /KiTS-00011/, KiTS-00122, KiTS-00170, KiTS-00197, KiTS-00206: size of segmentation and image differs slightly, so excluded

labels from
https://github.com/wukaiyeah/kidney_cancer_project
