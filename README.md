# Quickest detection of false data injection attack in remote state estimation

## Sample Path Generation
FDIGenerateDataNew.ipynb generates sample paths for optimizing the threshold value. Number of samples paths can be set using the variable `num_paths`, the horizon length using `N` and the probability of FDI attack at each step using `theta`.

## Optimizing for Threshold Gamma
SingleThresholdSGD.py finds the optimal values of the threshold and subject to different probability of false alarm (PFA) rates through an amalgamation of Simultaneous Pertubation Stochastic Approximation and Two-Timescale Approximation algorithms. Number of iterations for each PFA can be set using `num_iters` and the batch size using `batch_size`.

## Cite
[IEEE ISIT 2021 paper](https://ieeexplore.ieee.org/document/9518036):

```
A. Gupta, A. Sikdar and A. Chattopadhyay, 
"Quickest detection of false data injection attack in remote state estimation," 
2021 IEEE International Symposium on Information Theory (ISIT), 2021, pp. 3068-3073, 
doi: 10.1109/ISIT45174.2021.9518036.
```
