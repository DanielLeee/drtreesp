from scipy.optimize import minimize
import numpy as np
import pqn
import time
import mst
import torch
import util
import stochastic
import doubleoracle


def proj_simplex(orig_vector, radius=1.0):
    """
    https://github.com/spencerkent/resonator-networks/blob/master/resonator_networks/utils/constraint_projection.py
    This is the simplest simplex projection method. It's O(nlogn), but works fine
    Parameters
    ----------
    orig_vector : ndarray(size=(n,))
      The vector to be projected onto the simplex
    radius : float, optional
      The radius of the simplex, \sum_i |orig_vector_i| = radius. Default 1.0.
    """
    assert radius > 0, "requested radius of the simplex must be positive"
    ori_shape = orig_vector.shape
    orig_vector = orig_vector.reshape(-1)
    # check if orig_vector is already on the simplex, then we're done
    if np.sum(orig_vector) == radius and np.alltrue(orig_vector >= 0):
        return orig_vector
    sorted_htl = np.sort(orig_vector)[::-1]  # sorted _h_igh _t_o _l_ow
    cumulative_sum = np.cumsum(sorted_htl)
    rho = np.nonzero(sorted_htl > ((cumulative_sum - radius) /
                                 np.arange(1, len(orig_vector) + 1)))[0][-1]
    # in terms of index-starting-at-zero convention
    theta = (cumulative_sum[rho] - radius) / (rho + 1)

    return (orig_vector - theta).clip(min=0).reshape(ori_shape)


def u_k_dual_func_grad(model_params, k, w, n, num_rels):

    alpha = model_params
    alpha[k] = alpha.max() + 1
    alpha_repeat = np.repeat(np.expand_dims(alpha, axis = 0), n, axis = 0).T
    alpha_min = np.minimum(alpha_repeat, alpha_repeat.T)
    obj_val = -alpha[:k].sum() - alpha[k+1:].sum()
    obj_grad = -np.ones(n)
    obj_grad[k] = 0

    w_rel_alpha_min = w - num_rels * alpha_min
    above_thr = w_rel_alpha_min < 0
    below_thr = ~above_thr
    obj_val += (above_thr * np.square(w) / num_rels / 2 + below_thr * alpha_min * (w - num_rels * alpha_min / 2)).sum() / 2
    obj_grad += ((1 - np.eye(n)) * (alpha_repeat == alpha_min) * below_thr * w_rel_alpha_min).sum(1)

    obj_val *= -1
    obj_grad *= -1

    return [obj_val, obj_grad]


def proj_u_k(to_proj_w, k, n, num_rels):
    
    n_arcs = n * n * num_rels
    res = np.zeros((n, n, num_rels))
    w = to_proj_w.reshape(n, n, num_rels)
    if k == 0:
        res[:, 0] = 0
        for j in range(1, n):
            res[np.arange(n) != j, j] = proj_simplex(w[np.arange(n) != j, j])
    else:
        wsum = w.sum(axis = 2)
        w_agg = wsum + wsum.T 
        initial_params = np.random.rand(n)
        opt_params = minimize(u_k_dual_func_grad, initial_params, args = (k, w_agg, n, num_rels), method = 'L-BFGS-B', jac = True, options = {'disp' : 0})
        opt_alpha = opt_params.x
        opt_alpha_repeat = np.repeat(np.expand_dims(opt_alpha, 0), n, axis = 0)
        opt_alpha_min = np.minimum(opt_alpha_repeat, opt_alpha_repeat.T)
        res = (w - np.expand_dims(np.minimum(opt_alpha_min / 2, w_agg / num_rels / 2), axis = -1)) * (1 - np.expand_dims(np.eye(n), axis = -1))

    return res.reshape(-1)


def proj_arb_admm(to_proj_w, n, num_rels, eps_tol = 1e-2, tau_incr = 1.1, tau_decr = 1.1, mu = 1, max_iter = 100, record_history = False):

    n_arcs = n * n * num_rels
    rho = np.ones(n)
    lambd = np.zeros((n, n_arcs))
    
    mask = (1 - np.expand_dims(np.eye(n), axis = (0, -1)))
    u = (np.random.rand(n, n, n, num_rels) * mask).reshape(n, n_arcs)
    rho[0] = 0
    lambd[0] = 0
    
    if record_history:
        history = [np.square(to_proj_w - u[0]).sum()]

    for iter_idx in range(max_iter):
        for k in range(1, n):
            vec_to_proj_k = (2 * to_proj_w + rho[k] * n * (u[0] + lambd[k])) / (2 + rho[k] * n)
            u[k] = proj_u_k(vec_to_proj_k, k, n, num_rels)
        vec_to_proj_r = (2 * to_proj_w + rho.dot(u - lambd) * n) / (2 + rho.sum() * n)
        old_u_r = np.copy(u[0])
        u[0] = proj_u_k(vec_to_proj_r, 0, n, num_rels)
        lambd[1:] += u[0] - u[1:]

        # diff = np.abs(u[1:] - u[0:1]).reshape(n - 1, -1).sum(-1)
        # print(diff)

        if record_history:
            history.append(np.square(to_proj_w - u[0]).sum())

        primal_res = u[0] - u[1:]
        dual_res = np.outer(rho[1:], old_u_r - u[0])
        if (not record_history) and np.linalg.norm(primal_res) < eps_tol * max(np.linalg.norm(u[1:], axis = 1).sum(), np.linalg.norm(u[0]) * (n - 1)) and np.linalg.norm(dual_res) < eps_tol * np.linalg.norm(np.expand_dims(rho, axis = -1)* lambd, axis = 1).sum():
            break
        for k in range(1, n):
            norm_primal = np.linalg.norm(primal_res[k - 1])
            norm_dual = np.linalg.norm(dual_res[k - 1])
            if norm_primal > mu * norm_dual:
                rho[k] *= tau_incr
                lambd[k] /= tau_incr
            elif norm_dual > mu * norm_primal:
                rho[k] /= tau_decr
                lambd[k] *= tau_decr

    # print(np.square(to_proj_w - u[0]).sum())
    
    if record_history:
        return u[0], history
    else:
        return u[0]


def get_initial_legal_p_from_data(data, num_rels):

    ret = []
    for data_instance in data:
        n = data_instance['length']
        p_cur = np.zeros((n, n, num_rels))
        p_cur[np.arange(0, n - 1), np.arange(1, n), 0] = 1
        ret.append(p_cur.reshape(-1))
    ret = np.concatenate(ret)

    return ret


def adv_train_pqn(trian_data, eval_data, test_data, num_rels, mu, lambd):
    
    initial_params = get_initial_legal_p_from_data(data, num_rels)

    opt_p_check, _, _ = pqn.minConf_PQN(lambda x : adv_func_grad(x, data = data, num_rels = num_rels, mu = mu, lambd = lambd)[:2], initial_params, lambda x : proj_arb_set(x, data = data, num_rels = num_rels), verbose = 3, optTol = 1e-9)

    _, _, opt_theta, _ = adv_func_grad(opt_p_check, data, num_rels, mu, lambd)
    
    opts = [(0, 0, 0, 0), (0, 0, 0, 0), None]

    adv_theta_callback(opt_theta, eval_data, test_data, num_rels, mu, lambd, opts)

    return opt_theta, opts[1]


def proj_arb(w, n, num_rels, max_iter = 100, record_history = False):

    w = w.reshape(n, n, num_rels)
    x = mst.get_mst_np(-w)
    if record_history:
        history = [np.square(w - x).sum()]

    for t in range(max_iter):
        grad = 2 * (x - w)
        s = mst.get_mst_np(grad)
        if record_history:
            step_size = 2 / (t + 2)
        else:
            tem0 = ((s - x) * (w - x)).sum()
            tem1 = np.square(s - x).sum()
            if tem1 < 1e-9:
                break
            else:
                step_size = np.clip(tem0 / tem1, 0, 1)
        x += step_size * (s - x)
        if record_history:
            history.append(np.square(w - x).sum())

    x = x.reshape(-1)

    if record_history:
        return x, history
    else:
        return x


def proj_arb_set(to_proj_w, data, num_rels):

    n_data = len(data)
    p_check = to_proj_w
    p_proj = np.zeros(p_check.shape)

    start_idx = 0
    for idx, data_instance in enumerate(data):
        n = data_instance['length']
        num_arcs = data_instance['num_arcs']
        end_idx = start_idx + num_arcs
        p_check_idx = p_check[start_idx : end_idx]
        p_proj[start_idx : end_idx] = proj_arb(p_check_idx, n, num_rels)
        start_idx = end_idx
    
    return p_proj


def feature_dot(ori_feature, left_multiplier = None, right_multiplier = None, feature_type = 2):

    '''
    type 0 : (n * n * n_rels, n_feature)
    type 1 : (2, n, n_feature), dep -> head in the 1st dimension, requireing n_rels = 1, Hadamard product
    type 2 : (2, n, n_feature), dep -> head in the 1st dimension, requireing n_rels = 1, outer product
    '''

    if feature_type == 0:
        num_arcs, n_feature = ori_feature.shape
        if left_multiplier is not None:
            return left_multiplier.dot(ori_feature)
        elif right_multiplier is not None:
            return ori_feature.dot(right_multiplier)
        else:
            return ori_feature
    elif feature_type == 1:
        _, n, n_feature = ori_feature.shape
        if left_multiplier is not None:
            return np.einsum('ij,ik,jk->k', left_multiplier.reshape(n, n), ori_feature[0], ori_feature[1]).reshape(-1)
        elif right_multiplier is not None:
            return np.einsum('ik,jk,k->ij', ori_feature[0], ori_feature[1], right_multiplier).reshape(-1)
        else:
            return np.einsum('ik,jk->ijk', ori_feature[0], ori_feature[1]).reshape(n * n, -1)
    elif feature_type == 2:
        ori_feature = torch.tensor(ori_feature, dtype = torch.double)
        _, n, n_feature = ori_feature.shape
        if left_multiplier is not None:
            left_multiplier = torch.tensor(left_multiplier)
            return torch.einsum('ij,ik,jl->kl', left_multiplier.reshape(n, n), ori_feature[0], ori_feature[1]).reshape(-1).numpy()
        elif right_multiplier is not None:
            right_multiplier = torch.tensor(right_multiplier)
            return torch.einsum('ik,kl,jl->ij', ori_feature[0], right_multiplier.reshape(n_feature, n_feature), ori_feature[1]).reshape(-1).numpy()
        else:
            return torch.einsum('ik,jl->ijkl', ori_feature[0], ori_feature[1]).reshape(n * n, -1).numpy()


def adv_func_grad(model_params, data, num_rels, mu, lambd, given_theta = None, is_train = True):
    
    obj_val = 0.0
    obj_grad = np.zeros(model_params.size)
    p_check = model_params
    n_data = len(data)
    node_feature_dim = data[0]['feature'].shape[-1]
    n_feature = node_feature_dim ** 2
    p_hat = np.array(p_check)
    theta = np.zeros(n_feature)
    feature_diff = np.zeros(n_feature)

    if mu == 0:
        p_hat = mst.get_mst_set(-p_check, data, num_rels)
    else:
        p_hat = proj_arb_set(p_check / mu, data, num_rels)
    
    start_idx = 0
    for idx, data_instance in enumerate(data):
        
        num_arcs = data_instance['num_arcs']
        feature = data_instance['feature']
        p_data = data_instance['groundtruth']
        end_idx = start_idx + num_arcs
        p_check_idx = p_check[start_idx : end_idx]
        
        if is_train:
            feature_diff += feature_dot(feature, left_multiplier = p_check_idx - p_data)
        else:
            feature_diff += feature_dot(feature, left_multiplier = p_check_idx)

        start_idx = end_idx

    if given_theta is None:
        theta += feature_diff / (-n_data * lambd)
    else:
        theta += given_theta

    grad_theta = feature_diff / n_data + lambd * theta
    obj_val += (-p_hat.dot(p_check) + (np.square(p_hat).sum() - np.square(p_check).sum()) * mu * 0.5 + feature_diff.dot(theta)) / n_data + np.square(theta).sum() * lambd * 0.5
    obj_grad += - p_hat - mu * p_check

    start_idx = 0
    for idx, data_instance in enumerate(data):
        
        num_arcs = data_instance['num_arcs']
        feature = data_instance['feature']
        end_idx = start_idx + num_arcs
        
        obj_grad[start_idx : end_idx] += feature_dot(feature, right_multiplier = theta)

        start_idx = end_idx

    obj_val *= -1
    obj_grad *= -1 / n_data

    if given_theta is None:
        return (obj_val, obj_grad, theta, grad_theta)
    else:
        return (obj_val, obj_grad, p_hat, grad_theta)


def adv_train(train_data, eval_data, test_data, num_rels, mu, lambd, max_iter = 1000):
    
    p_check = get_initial_legal_p_from_data(train_data, num_rels)
    opts = [(0, 0, 0, 0), (0, 0, 0, 0), None]
    inf_method = get_inf_method('marginal', False, num_rels, mu, lambd)

    for t in range(max_iter):
        
        obj_val, grads, theta, _ = adv_func_grad(p_check, train_data, num_rels, mu, lambd)
        s = mst.get_mst_set(grads, train_data, num_rels)
        step_size = 2 / (t + 2)
        print('iter = {}, step = {:.8f}, obj = {}'.format(t, step_size, obj_val))
        p_check += step_size * (s - p_check)
        adv_theta_callback(theta, eval_data, test_data, num_rels, mu, lambd, opts, inf_method)
    
    return opts[2], opts[1]


def adv_inference(data, num_rels, mu, lambd, theta, probabilistic = False, max_iter = 100):
     
    if probabilistic:
        _, _, p_hat = adv_theta_func_grad(theta, data, num_rels, mu, lambd, max_iter = max_iter, is_train = False)
        w = -p_hat
    else:
        w = []
        for idx, data_instance in enumerate(data):
            feature = data_instance['feature']
            w.append(feature_dot(feature, right_multiplier = theta))
        w = np.concatenate(w)
        w = -w

    return mst.get_mst_set(w, data, num_rels)
    

def adv_theta_func_grad(theta, data, num_rels, mu, lambd, max_iter = 10, is_train = True, is_batch = False, batch_size = 128):

    if is_batch:
        data = np.random.choice(data, batch_size, False)

    p_check = get_initial_legal_p_from_data(data, num_rels)
    opt_obj_val = float('inf')
    opt_grad_theta = None
    opt_p_hat = None

    for t in range(max_iter):
        obj_val, grad_p_check, p_hat, grad_theta = adv_func_grad(p_check, data, num_rels, mu, lambd, given_theta = theta, is_train = is_train)
        if obj_val < opt_obj_val:
            opt_obj_val = obj_val
            opt_grad_theta = grad_theta
            opt_p_hat = p_hat
        s = mst.get_mst_set(grad_p_check, data, num_rels)
        step_size = 2 / (t + 2)
        p_check += step_size * (s - p_check)
        # print('iter = {}, step = {:.8f}, obj = {}'.format(t, step_size, obj_val))

    opt_obj_val *= -1
    
    return [opt_obj_val, opt_grad_theta, opt_p_hat]


def adv_theta_callback(theta, eval_data, test_data, num_rels, mu, lambd, opts, inf_method):

    opt_eval, opt_test, opt_ret = opts

    eval_preds = inf_method(eval_data, theta)
    cur_eval = util.eval_set(eval_data, num_rels, mu, lambd, theta, eval_preds)
    if cur_eval[0] > opt_eval[0]:
        opt_eval = cur_eval
        opt_ret = theta
        test_preds = inf_method(test_data, theta)
        opt_test = util.eval_set(test_data, num_rels, mu, lambd, theta, test_preds)

    print('Current eval: UAS = {:.4f}, LAS = {:.4f}, UCM = {:.4f}, LCM = {:.4f}'.format(*cur_eval))
    print('Optimal eval: UAS = {:.4f}, LAS = {:.4f}, UCM = {:.4f}, LCM = {:.4f}'.format(*opt_eval))
    print('Optimal test: UAS = {:.4f}, LAS = {:.4f}, UCM = {:.4f}, LCM = {:.4f}'.format(*opt_test))
    print()

    opts[0] = opt_eval
    opts[1] = opt_test
    opts[2] = opt_ret

    return


def get_inf_method(adv_method, prob_inf, num_rels, mu, lambd):

    assert(adv_method in ['marginal', 'game'])
    assert(prob_inf in [True, False])
    
    if adv_method == 'marginal' or not prob_inf:
        inf_method = lambda x, y : adv_inference(data = x, num_rels = num_rels, mu = mu, lambd = lambd, theta = y, probabilistic = prob_inf)
    else:
        inf_method = lambda x, y : doubleoracle.adv_game_inference(data = x, num_rels = num_rels, lambd = lambd, theta = y)
    
    return inf_method


def adv_stochastic_train(train_data, eval_data, test_data, num_rels, mu, lambd, batch_size, adv_method, prob_inf, **kwargs):
    
    assert(adv_method in ['marginal', 'game'])

    node_feature_dim = train_data[0]['feature'].shape[-1]
    n_feature = node_feature_dim ** 2
    initial_theta = np.random.rand(n_feature)
    opts = [(0, 0, 0, 0), (0, 0, 0, 0), None]

    opt_fun = stochastic.adam

    inf_method = get_inf_method(adv_method, prob_inf, num_rels, mu, lambd)

    if adv_method == 'marginal':
        func_grad_x = lambda x : adv_theta_func_grad(x, data = train_data, num_rels = num_rels, mu = mu, lambd = lambd, is_batch = True, batch_size = batch_size)[:2]
    elif adv_method == 'game':
        func_grad_x = lambda x : doubleoracle.adv_game_train_func_grad(x, data = train_data, num_rels = num_rels, lambd = lambd, is_batch = True, batch_size = batch_size)

    callback_x = lambda x : adv_theta_callback(x, eval_data = eval_data, test_data = test_data, num_rels = num_rels, mu = mu, lambd = lambd, opts = opts, inf_method = inf_method)

    opt_res = opt_fun(func_grad = func_grad_x, x0 = initial_theta, callback = callback_x, **kwargs)
    opt_theta = opt_res.x

    return opt_theta, opts[1]

