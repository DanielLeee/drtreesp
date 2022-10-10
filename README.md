# Moment Distributionally Robust Tree Structured Prediction

This is the official implementation of the following paper accepted to *NeurIPS 2022*:

> **Moment Distributionally Robust Tree Structured Prediction**
> 
> Yeshu Li, Danyal Saeed, Xinhua Zhang, Brian D. Ziebart, Kevin Gimpel
> 
> [[Proceeding PDF link TBA]]() [[Virtual]](https://nips.cc/virtual/2022/poster/54233) [[OpenReview]](https://openreview.net/forum?id=Tq2XqINV1Jz)

## Requirements

- numpy
- scipy
- torch
- cffi
- gcc
- supar

## Installation

Run

```shell
python cmst_extension_build.py
```

to compile the C implementation of [Fast MST](https://github.com/stanojevic/Fast-MST-Algorithm) for finding the minimum weight directed spanning tree in the worst-case time complexity of $\mathcal{O}(n^2)$.

## Data Preparation



## Quick Start

Change the data path and algorithm to run in `exp.py`.

Run

```shell
python exp.py
```

## End-to-end Training



## Citation

Please cite our work if you find it useful in your research:

```
TO DO
```

## Acknowledgement

This project is based upon work supported by the National Science Foundation under Grant No. 1652530.

Part of the code is based on [SuPar](https://github.com/yzhangcs/parser), [HPSG](https://github.com/DoodleJZ/HPSG-Neural-Parser), [zmsp](https://github.com/zzsfornlp/zmsp), [Fast MST](https://github.com/stanojevic/Fast-MST-Algorithm).
