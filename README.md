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

Refer to the [released data](https://github.com/DanielLeee/drtreesp/releases/download/data/data.zip) for the complete datasets and train/val/test splits in the paper and some examples of the processed data as inputs of our methods. See `script/supar_exp.ipynb` for reproducing baseline experiments and generating input data with pre-trained features for our methods.

Namely, we adopt the data format as follows:

```Python REPL
>>> data.keys()
dict_keys(['train', 'dev', 'test'])
>>> len(data['train'])
100
>>> data['train'][0].keys()
dict_keys(['length', 'sentence', 'feature', 'groundtruth', 'punctuation_mask', 'relation_prediction_all_arcs'])
>>> data['train'][0]['length']
45 # dummy root + number of tokens
>>> data['train'][0]['sentence']
"Yes , he says , premiums on such variable-rate coverage can be structured to `` vanish '' after a certain period -- but usually only if interest rates stay high enough to generate sufficient cash to cover the annual cost of insurance protection ."
>>> data['train'][0]['feature'].shape
(2, 45, 500) # [head + dependent, total length, node-wise feature dimension]
>>> data['train'][0]['groundtruth'].shape
(44, 2) # [number of tokens, arc prediction + relation prediction]
>>> data['train'][0]['punctuation_mask'].shape
(44,) # [number of tokens]
>>> data['train'][0]['relation_prediction_all_arcs'].shape
(45, 45) # [total length, total length]
```

## Quick Start

Change the data path and algorithm to run in `exp.py`.

Run

```shell
python exp.py
```

## End-to-end Training

Use `BiaffineDependencyModel` in [SuPar](https://github.com/yzhangcs/parser) as an example:

```Python
model = BiaffineDependencyModel()
criterion = advneural.AdvLoss()
s_arc, s_rel = model(input)
# s_arc: [batch_size, max_len, max_len] (arc-wise score matrices)
# arcs: [batch_size, max_len] (arc heads groundtruth)
# s_rel: [batch_size, max_len, max_len, num_rels] (relation score matrices)
# rels: [batch_size, max_len] (relation label groundtruth)
# mask: [batch_size, max_len] (mask for a batch of sentences with various lengths)
loss = criterion(s_arc, arcs, s_rel, rels, mask, lambd)
loss.backward()
```

`advneural.py` serves as an example and is not optimized for applications yet.

## Citation

Please cite our work if you find it useful in your research:

```
TO DO
```

## Acknowledgement

This project is based upon work supported by the National Science Foundation under Grant No. 1652530.

Part of the code is based on [SuPar](https://github.com/yzhangcs/parser), [HPSG](https://github.com/DoodleJZ/HPSG-Neural-Parser), [zmsp](https://github.com/zzsfornlp/zmsp), [Fast MST](https://github.com/stanojevic/Fast-MST-Algorithm).
