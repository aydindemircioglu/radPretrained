
# Predictive performance of radiomic models based on features extracted from pretrained deep networks


This is the code for the paper ['Predictive performance of radiomic models based on features extracted from pretrained deep networks'](https://doi.org/10.1186/s13244-022-01328-y).


## Requirements

Although there is a requirements file, no priority is given to reach 100%
reproducibility. Apart from theoretical considerations, for example, fixing
the torch version with an older cuda seems pointless.
However, SimpleITK == 2.0.2 is needed, because later version will give a few
"no orthonormal definition found" errors, at least on CRLM-004 and CRLM-063.
Also, for AUC computations and DeLong tests, a full R environment is needed.


## Data

Make sure all WORC images/masks exist. They can be downloaded using the XNA
online tool. More infos in the publication from Starmans et al.
The pathes can be modified in the parameters.py. Please also copy the HN
dataset (which is somewhat different) into WORC/HN. So the folders look
(yes, not nice)

```
worc/ -- Melanoma_001 -- <DCM files>
      -- CRLM_001 -- <DCM files>
      -- HM -- HM_1104 -- <DCM files>
```

## Create resampled data

We have to first resample all scans, so we first create those.

./resampleData.py



## Extract features

Features are extracted before the ML experiment, both, for deep and generic models.
Just call ./extractFeatures.py



## Experiment

After feature extraction, just start the experiment with
./startExperiment.py



## Evaluation

Evaluation proceeds with ./evluate.py. It will create files in results/*
and paper/*, latter are the figures in the paper (only paper, not supplementary)


## Notes

The code is neither overly polished nor optimized. It also contains some parts
that come from older projects and which I decided not to use (e.g. PR-curves)--
a few of these things were computed during the experiment, so are also part
of the results file. There are a few fixed to removed these during evaluation.


## Licence

Note: The data has its own license. Please refer to the respective publications for more information.

Other code is licensed with the MIT license:

Copyright (c) 2022 Aydin Demircioglu Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
