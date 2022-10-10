from scipy.optimize import linprog
import numpy as np
import mst
import advtree
import time


def play_one_adv_game(feature, feature_dot_theta, n, num_rels, num_arcs, eps = 1e-2, max_iter = 200):

    pred_set = np.zeros((1, n, n, num_rels))
    pred_set[0, np.arange(0, n - 1), np.arange(1, n), 0] = 1
    pred_set = pred_set.reshape(1, num_arcs)
    adv_set = np.zeros((1, n, n, num_rels))
    adv_set[0, np.arange(0, n - 1), np.arange(1, n), 0] = 1
    adv_set = adv_set.reshape(1, num_arcs)
    phi = feature_dot_theta
    v = np.zeros(4)

    for iter_idx in range(max_iter):

        updated = False

        num_adv = adv_set.shape[0]
        num_pred = pred_set.shape[0]
        mat_adv_pred = -adv_set.dot(pred_set.T)
        adv_phi = adv_set.dot(phi)
        mat_adv_pred += adv_phi.reshape(num_adv, 1)
        minus_row = -np.ones((num_adv, 1))
        A_ub = np.concatenate((mat_adv_pred, minus_row), axis = 1)
        b_ub = np.zeros(num_adv)
        A_eq = np.ones((1, num_pred + 1))
        A_eq[0, -1] = 0
        b_eq = np.ones(1)
        bounds = [(0, None)] * num_pred + [(None, None)]
        c = (1 - A_eq).reshape(-1)
        # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, options = {'cholesky' : False, 'sym_pos' : False, 'lstsq' : True})
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
        pred_dist = res.x[:-1]
        v[0] = res.fun

        w = -pred_dist.dot(pred_set) + phi
        br_adv = mst.get_mst_np(-w.reshape(n, n, num_rels)).reshape(1, num_arcs)
        v[1] = br_adv.dot(w)
        if abs(v[0] - v[1]) > eps: 
            updated = True
            adv_set = np.concatenate((adv_set, br_adv), axis = 0)

        num_pred = pred_set.shape[0]
        num_adv = adv_set.shape[0]
        mat_pred_adv = -pred_set.dot(adv_set.T)
        adv_phi = adv_set.dot(phi)
        mat_pred_adv += adv_phi.reshape(1, num_adv)
        minus_row = -np.ones((num_pred, 1))
        A_ub = -np.concatenate((mat_pred_adv, minus_row), axis = 1)
        b_ub = np.zeros(num_pred)
        A_eq = np.ones((1, num_adv + 1))
        A_eq[0, -1] = 0
        b_eq = np.ones(1)
        bounds = [(0, None)] * num_adv + [(None, None)]
        c = (A_eq - 1).reshape(-1)
        # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, options = {'cholesky' : False, 'sym_pos' : False, 'lstsq' : True})
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
        adv_dist = res.x[:-1]
        v[2] = -res.fun

        w = -adv_dist.dot(adv_set)
        br_pred = mst.get_mst_np(w.reshape(n, n, num_rels)).reshape(1, num_arcs)
        v[3] = br_pred.dot(w) - w.dot(phi) 
        if abs(v[2] - v[3]) > eps: 
            updated = True
            if iter_idx == max_iter - 1:
                break
            pred_set = np.concatenate((pred_set, br_pred), axis = 0)

        # print(v)
        # print(pred_set.shape[0], adv_set.shape[0])

        if not updated:
            break

    pred_marginal = pred_dist.dot(pred_set)
    adv_marginal = adv_dist.dot(adv_set)
    grad_theta = advtree.feature_dot(feature, left_multiplier = adv_marginal)

    return v[0], grad_theta, pred_marginal


def adv_game_train_func_grad(model_params, data, num_rels, lambd, is_batch, batch_size):

    if is_batch:
        data = np.random.choice(data, batch_size, False)

    n_data = len(data)
    theta = model_params
    obj_val = 0.0
    obj_grad = np.zeros(theta.shape)


    for idx, data_instance in enumerate(data):
        n = data_instance['length']
        feature = data_instance['feature']
        p_data = data_instance['groundtruth']
        num_arcs = data_instance['num_arcs']
        feature_dot_theta = advtree.feature_dot(feature, right_multiplier = theta)
        game_val, game_grad, _ = play_one_adv_game(feature, feature_dot_theta, n, num_rels, num_arcs)
        obj_val += game_val - p_data.dot(feature_dot_theta)
        obj_grad += game_grad - advtree.feature_dot(feature, left_multiplier = p_data)
    
    obj_val /= n_data
    obj_val += np.square(theta).sum() * lambd * 0.5

    obj_grad /= n_data
    obj_grad += lambd * theta

    return (obj_val, obj_grad)


def adv_game_inference(data, num_rels, lambd, theta):

    total_arcs = data[-1]['end_idx'] + 1
    pred = np.zeros(total_arcs)
    for idx, data_instance in enumerate(data):
        start_idx = data_instance['start_idx']
        end_idx = data_instance['end_idx']
        feature = data_instance['feature']
        n = data_instance['length']
        num_arcs = data_instance['num_arcs']
        feature_dot_theta = advtree.feature_dot(feature, right_multiplier = theta)
        _, _, pred_idx = play_one_adv_game(feature, feature_dot_theta, n, num_rels, num_arcs)
        pred[start_idx : end_idx] = pred_idx
    w = -pred
    hard_pred = mst.get_mst_set(w, data, num_rels)

    return hard_pred

