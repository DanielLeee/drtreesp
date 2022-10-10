'''
Adapted from Miloš Stanojević's code here: https://github.com/stanojevic/Fast-MST-Algorithm/blob/main/mst.py
'''
from _cmst import ffi, lib
import numpy as np
import time
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# convention: [from, to, relation]
def get_mst_np(np_arr, one_root = True):

    n, n, num_rels = np_arr.shape
    np_arr_size = np_arr.size
    w_contiguous = np.ascontiguousarray(np_arr.astype(np.float64))
    result_pointer = ffi.new('int32_t [{}]'.format(np_arr_size))
    weight_pointer = ffi.from_buffer('double *', w_contiguous)
    one_root_cast = ffi.cast('bool', one_root)
    n_cast = ffi.cast('int32_t', n)
    num_rels_cast = ffi.cast('int32_t', num_rels)
    lib.fast_parse(weight_pointer, one_root_cast, n_cast, num_rels_cast, result_pointer)
    ret_mat = np.frombuffer(ffi.buffer(result_pointer, np_arr_size * 4), dtype = np.int32).reshape(n, n, num_rels).astype(np.float64)

    return ret_mat


def get_mst_set(target, data, num_rels):

    ret = np.zeros(target.size)
    start_idx = 0
    for idx, data_instance in enumerate(data):
        n = data_instance['length']
        num_arcs = data_instance['num_arcs']
        end_idx = start_idx + num_arcs
        ret[start_idx : end_idx] = get_mst_np(target[start_idx : end_idx].reshape(n, n, num_rels)).reshape(-1)
        start_idx = end_idx

    return ret


def get_mst_set_by_lens(target, lens):

    target = target.detach().cpu().numpy()
    ret = np.zeros(target.size)
    start_idx = 0
    for n in lens:
        end_idx = start_idx + n * n
        ret[start_idx : end_idx] = get_mst_np(target[start_idx : end_idx].reshape(n, n, 1)).reshape(-1)
        start_idx = end_idx
    ret = torch.tensor(ret, dtype = torch.float32).to(device)

    return ret

