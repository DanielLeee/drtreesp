import numpy as np


def gt_to_prob_vector(groundtruth):

    n = groundtruth.shape[0] + 1
    p = np.zeros((n, n))
    gt = np.array(groundtruth)
    p[gt[:, 0], np.arange(1, n)] = 1
    p = p.reshape(-1)

    return p


def preprocess_data(data, num_rels):

    start_idx = 0
    # n_feature = 50
    for idx, data_instance in enumerate(data):
        n = data_instance['length']
        # data_instance['feature'] = data_instance['feature'].reshape(n * n * num_rels, n_feature).astype(np.float32)
        data_instance['feature'] = np.concatenate((data_instance['feature'], np.ones((2, n, 1))), axis = -1, dtype = np.float32) / np.sqrt(500)
        data_instance['original_groundtruth'] = data_instance['groundtruth']
        data_instance['groundtruth'] = gt_to_prob_vector(data_instance['groundtruth'])
        data_instance['num_arcs'] = n * n * num_rels
        start_idx += n * n * num_rels

    return 0


def eval_set(data, num_rels, mu, lambd, opt_theta, given_prediction):
    
    hard_pred = given_prediction
    las = 0
    uas = 0
    lcm = 0
    ucm = 0
    n_tokens = 0
    start_idx = 0
    for idx, data_instance in enumerate(data):
        n = data_instance['length']
        num_arcs = data_instance['num_arcs']
        relation_all = data_instance['relation_prediction_all_arcs']
        punct_mask = data_instance['punctuation_mask']
        gt = data_instance['original_groundtruth']
        end_idx = start_idx + num_arcs
        arc_preds = hard_pred[start_idx : end_idx].reshape(n, n).argmax(0)[1:]
        rel_preds = relation_all[arc_preds, np.arange(1, n)]
        arc_mask = (arc_preds == gt[:, 0]) & punct_mask
        uas += arc_mask.sum()
        las += ((rel_preds == gt[:, 1]) & arc_mask).sum()
        ucm += int(arc_mask.sum() == punct_mask.sum())
        lcm += int((rel_preds == gt[:, 1])[arc_mask].sum() == punct_mask.sum())
        n_tokens += punct_mask.sum()
        start_idx = end_idx
    las /= n_tokens
    uas /= n_tokens
    lcm /= len(data)
    ucm /= len(data)
    # print('UAS = {:.4f}, LAS = {:.4f}, UCM = {:.4f}, LCM = {:.4f}'.format(uas, las, ucm, lcm))

    return uas, las, ucm, lcm

