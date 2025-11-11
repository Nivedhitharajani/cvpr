# CVPR Coursework

This MATLAB project performs color-based image retrieval tasks to investigate the effect of various distance measures on the comparability among the whole RGB color, grid-based RGB color, and HS color histograms. The precision recall study, mean average precision (mAP), and various confusion matrices are conducted to assess accuracy, robustness, and efficiency. The outcome confirms that illumination variations are better taken care of by HS features, with greater spatial awareness by grid-based RGB features and excellent compression by PCA on HS features.

## Structure
```
CVPR_Coursework/
  data/   # MSRCv2 dataset 
  desc/   # saved descriptors
  figs/   # PR curves & screenshots
  src/    # MATLAB source code
```

## How to run (after you add code)
```matlab
addpath(genpath('src'));
% cvpr_computedescriptors('desc','color8');
% cvpr_visualsearch('desc','color8','metric','CHI2');
% evaluate_search('color8','CHI2',20);
```
