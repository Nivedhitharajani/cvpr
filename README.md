# CVPR Coursework

## Structure
```
CVPR_Coursework/
  data/   # MSRCv2 dataset (not tracked)
  desc/   # saved descriptors (not tracked)
  figs/   # PR curves & screenshots (not tracked)
  src/    # MATLAB source code
```

## How to run (after you add code)
```matlab
addpath(genpath('src'));
% cvpr_computedescriptors('desc','color8');
% cvpr_visualsearch('desc','color8','metric','CHI2');
% evaluate_search('color8','CHI2',20);
```
