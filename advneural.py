import numpy as np
import mst
import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def proj_arb(w, n, num_rels, max_iter = 100):

    w = w.reshape(n, n, num_rels)
    x = mst.get_mst_np(-w.detach().cpu().numpy())
    x = torch.tensor(x, dtype = torch.float32).to(device)

    for t in range(max_iter):
        grad = 2 * (x - w)
        s = mst.get_mst_np(grad.detach().cpu().numpy())
        s = torch.tensor(s, dtype = torch.float32).to(device)
        tem0 = ((s - x) * (w - x)).sum()
        tem1 = torch.square(s - x).sum()
        if tem1 < 1e-9:
            break
        else:
            step_size = torch.clip(tem0 / tem1, 0, 1)
        x += step_size * (s - x)

    x = x.reshape(-1)

    return x


def proj_arb_set(to_proj_w, lens):

    p_check = to_proj_w
    p_proj = torch.zeros(p_check.shape, dtype = torch.float32).to(device)

    start_idx = 0
    for i, n in enumerate(lens):
        end_idx = start_idx + n * n
        p_check_idx = p_check[start_idx : end_idx]
        p_proj[start_idx : end_idx] = proj_arb(p_check_idx, n, 1)
        start_idx = end_idx
    
    return p_proj


def adv_func_grad(p_check, feats, lens, mu, lambd):
    
    obj_grad = torch.zeros(p_check.shape).to(device)
    n_data = feats.shape[0]

    if mu == 0:
        p_hat = mst.get_mst_set_by_lens(-p_check, lens)
    else:
        p_hat = proj_arb_set(p_check / mu, lens)

    obj_grad += - p_hat - mu * p_check
    p_check_dot_feat = 0.0
    start_idx = 0
    for i, n in enumerate(lens):
        end_idx = start_idx + n * n
        obj_grad[start_idx : end_idx] += feats[i, :n, :n].reshape(-1)
        p_check_dot_feat += feats[i, :n, :n].reshape(-1).dot(p_check[start_idx : end_idx])
        start_idx = end_idx

    obj_val = (-p_hat.dot(p_check) + (torch.square(p_hat).sum() - torch.square(p_check).sum()) * mu * 0.5 + p_check_dot_feat) / (-n_data)
    obj_grad *= -1 / n_data

    return obj_val, obj_grad


def get_initial_legal_p_from_data(lens):

    ret = []
    for n in lens:
        p_cur = torch.zeros((n, n)).to(device)
        p_cur[torch.arange(0, n - 1), torch.arange(1, n)] = 1
        ret.append(p_cur.reshape(-1))
    ret = torch.cat(ret)

    return ret


def gt_to_prob_vector(gt):

    n = gt.shape[0] + 1
    p = torch.zeros(n, n).to(device)
    p[gt, torch.arange(1, n)] = 1

    return p


def rel_to_prob_vector(head, rel, num_rels):
    
    n = head.shape[0] + 1
    p = torch.zeros(n, n, num_rels).to(device)
    p[head, torch.arange(1, n), rel] = 1

    return p


def adv_neural_train(s_arc, arcs, s_rel, rels, mask, lambd, mu = 0, max_iter = 100):

    s_arc = s_arc.permute(0, 2, 1)
    s_rel = s_rel.permute(0, 2, 1, 3)

    s_rel = s_arc.unsqueeze(-1) + s_rel
    s_arc, _ = s_rel.max(-1)

    n_data, _, _, num_rels = s_rel.shape
    lens = mask.sum(-1) + 1
    n_tokens = mask.sum()
    
    p_check = get_initial_legal_p_from_data(lens)
    opt_obj_val = torch.inf
    opt_p_check = None
    for t in range(max_iter):
        obj_val, grad_p_check = adv_func_grad(p_check, s_arc, lens, mu, lambd)
        if obj_val < opt_obj_val:
            opt_obj_val = obj_val.detach().clone()
            opt_p_check = p_check.detach().clone()
        s = mst.get_mst_set_by_lens(grad_p_check, lens)
        step_size = 2 / (t + 2)
        p_check += step_size * (s - p_check)

    loss = torch.tensor(0.0).to(device)
    start_idx = 0
    for i, n in enumerate(lens):
        end_idx = start_idx + n * n
        p_arc_data = gt_to_prob_vector(arcs[i][mask[i]])
        p_rel_data = rel_to_prob_vector(arcs[i][mask[i]], rels[i][mask[i]], num_rels)
        p_arc_adv = opt_p_check[start_idx : end_idx].reshape(n, n)
        p_rel_adv = F.one_hot((1 - p_rel_data + s_rel[i, :n, :n]).argmax(-1), num_classes = num_rels).detach() * p_arc_adv.unsqueeze(-1)
        # loss += ((p_arc_adv - p_arc_data) * s_arc[i, :n, :n]).sum()
        loss += ((p_rel_adv - p_rel_data) * s_rel[i, :n, :n]).sum()
        # loss += (torch.gather(s_rel[i, :n, :n], -1, (1 - p_rel_data + s_rel[i, :n, :n]).argmax(-1, keepdim = True)).squeeze() * p_arc_adv).sum()
        # loss -= s_rel[i][arcs[i][mask[i]], torch.arange(1, n), rels[i][mask[i]]].sum()

        # t = mst.get_mst_np(np.random.rand(n, n, 1))
        # t = mst.get_mst_np((-feats[i, :n, :n]).detach().cpu().numpy().reshape(n, n, 1))
        # t = torch.tensor(t, dtype = torch.float32).to(device).sum(-1)
        # opt_grad[i, :n, :n] = t - p_data
        # opt_grad[i, :n, :n] = - p_data
        start_idx = end_idx

    loss /= n_data
    # loss /= n_tokens

    # torch.set_printoptions(profile = "full")
    # print(feats[0][1])
    # torch.set_printoptions(profile = "default")
    
    return loss


def adv_neural_decode(feats, mask):
     
    feats = feats.permute(0, 2, 1)
    
    lens = mask.sum(-1) + 1
    arc_preds = torch.zeros(mask.shape, dtype = int).to(device)
    for i, n in enumerate(lens):
        pred = mst.get_mst_np(-feats[i, :n, :n].reshape(n, n, 1).detach().cpu().numpy())
        pred = torch.tensor(pred, dtype = torch.float32).to(device)
        pred = pred.sum(-1).argmax(0)[1:]
        arc_preds[i][mask[i]] = pred

    return arc_preds


class AdvLoss(nn.Module):

    def __init__(self):

        super().__init__()

        return

    def forward(self, s_arc, arcs, s_rel, rels, mask, lambd):
        
        new_mask = mask.detach().clone().to(torch.bool)
        new_mask[:, 0] = False

        loss = adv_neural_train(s_arc, arcs, s_rel, rels, new_mask, lambd)

        return loss

