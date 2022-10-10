from itertools import chain, combinations
import supar
import numpy as np
import advtree
import time
import torch
import advneural


def powerset(iterable):

    s = list(iterable)

    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def print_data_statistics(data):

    mean_val = 0.0
    min_val = float('inf')
    max_val = float('-inf')
    num_elements = 0

    for ele in data:
        mean_val += ele['feature'].sum()
        min_val = min(min_val, ele['feature'].min())
        max_val = max(max_val, ele['feature'].max())
        num_elements += ele['feature'].size

    mean_val /= num_elements

    print('min: {}'.format(min_val))
    print('max: {}'.format(max_val))
    print('mean: {}'.format(mean_val))

    return 0


def check_legal_tree(x):

    x = x.sum(-1)
    n = x.shape[0]

    ind = x.sum(0)
    ind_gt = np.ones(n)
    ind_gt[0] = 0
    error_ind = np.abs(ind - ind_gt)

    powers = powerset(np.arange(n))
    error_pow = np.zeros(len(list(powers)))
    for idx, subset in enumerate(powers):
        sub_mat = x[list(subset)][:, list(subset)]
        error_pow[idx] = max(0, sub_mat.sum() - len(subset) + 1)

    print(error_ind.max())
    print(error_ind)
    print(error_pow.max())
    print(error_pow)
    
    return



if __name__ == '__main__':

    
    '''
    # model = supar.Parser.load('biaffine-dep-en')
    model = supar.BiaffineDependencyParser.build('biaffine-dep-en')
    # loss, metric, processed_data_list = model.evaluate('data/ptb_sd330/test.conllx', verbose = True)
    # print(loss, metric, processed_data_list)
    model.train('data/ptb_sd330/train.trial0.subset1000.conllx', 'data/ptb_sd330/dev.conllx', 'data/ptb_sd330/test.conllx')
    '''
    

    '''
    data = np.load('data/ptb_sd330/ptb_arc_biaffine.npy', allow_pickle = True).item()['train']
    print_data_statistics(data)
    data = np.load('data/ptb_sd330/ptb_arc_biaffine_roberta.npy', allow_pickle = True).item()['train']
    print_data_statistics(data)
    data = np.load('data/ptb_sd330/ptb_arc_crf2o.npy', allow_pickle = True).item()['train']
    print_data_statistics(data)
    '''


    '''
    max_iter = 1000
    n = 8
    num_rels = 3
    to_proj_w = np.random.rand(n * n * num_rels) * 10 - 5
    t0 = time.time()
    res_admm, history_admm = advtree.proj_arb_admm(to_proj_w, n, num_rels, max_iter = max_iter)
    t1 = time.time()
    print(t1 - t0)
    res_fw, history_fw = advtree.proj_arb(to_proj_w, n, num_rels, max_iter = max_iter)
    t2 = time.time()
    print(t2 - t1)
    print(history_admm[-1])
    print(history_fw[-1])
    check_legal_tree(res_admm.reshape(n, n, num_rels))
    check_legal_tree(res_fw.reshape(n, n, num_rels))
    '''


    max_iter = 1000
    n = 5
    num_rels = 3
    num_runs = 100
    print('n = {}, num_rels = {}'.format(n, num_rels))
    beat_cnt = 0
    all_errors = []
    for i in range(num_runs):
        to_proj_w = np.random.rand(n * n * num_rels) * 10 - 5
        _, history_admm = advtree.proj_arb_admm(to_proj_w, n, num_rels, max_iter = max_iter, record_history = True)
        _, history_fw = advtree.proj_arb(to_proj_w, n, num_rels, max_iter = max_iter, record_history = True)
        error_admm = np.array(history_admm).reshape(1, max_iter + 1)
        error_fw = np.array(history_fw).reshape(1, max_iter + 1)
        errors = np.concatenate((error_admm, error_fw)).T
        all_errors.append(errors)
        if errors[-1, 0] < errors[-1, 1]:
            beat_cnt += 1
    print('win rate: {} / {}'.format(beat_cnt, num_runs))
    all_errors = np.array(all_errors)
    np.save('data/admmvsfw.npy', all_errors)


    '''
    n_data = 10
    max_len = 50
    lambd = 1e-4
    feats = np.random.rand(n_data, max_len, max_len)
    mask = np.zeros((n_data, max_len), dtype = bool)
    arcs = np.zeros((n_data, max_len), dtype = int)
    feats = torch.tensor(feats, dtype = torch.float32)
    mask = torch.tensor(mask)
    arcs = torch.tensor(arcs)

    for i in range(n_data):
        n = np.random.randint(5, 20)
        mask[i][1:n] = True
        arcs[i][1:n] = torch.tensor(np.random.randint(0, n, size = n - 1))

    obj_val, obj_grad = advneural.adv_neural_train(feats, arcs, mask, lambd)
    
    arc_preds = advneural.adv_neural_decode(feats, mask)

    m = np.random.rand(5, 5)
    '''


