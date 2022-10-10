import numpy as np
import advtree
import util


def multi_exps():

    metric_list = []
    for trial_idx in range(5):
        for n_train in [10, 50, 100, 1000]:
            ptb_data = np.load('data/ptb_sd330/ptb_biaffine_roberta_pretrainall_trial{}_subset{}.npy'.format(trial_idx, n_train), allow_pickle = True)
            ptb_data = ptb_data.item()
            num_rels = -1
            mu = 0
            lambd = 0.01 / n_train
            print('mu = {}, lambd = {}'.format(mu, lambd))

            for set_name in ptb_data:
                util.preprocess_data(ptb_data[set_name], num_rels = 1)
            
            train_data = ptb_data['train']
            eval_data = ptb_data['dev']
            test_data = ptb_data['test']
            print('#train : {}, #eval : {}, #test : {} '.format(len(train_data), len(eval_data), len(test_data)))
            
            print('train...')
            opt_theta, best_test_metric = advtree.adv_train(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd)
            # opt_theta, best_test_metric = advtree.adv_train_pqn(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd)
            # opt_theta, best_test_metric = advtree.adv_stochastic_train(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd, batch_size = min(200, n_train), adv_method = 'marginal', prob_inf = False, learning_rate = 0.1)
            # opt_theta, best_test_metric = advtree.adv_stochastic_train(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd, batch_size = min(10, n_train), adv_method = 'game', prob_inf = False, learning_rate = 0.1, maxiter = 500)
            metric_list.append(best_test_metric)

    for metric in metric_list:
        metric = [ele * 100 for ele in metric]
        print('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(*metric))

    return 0


def main():

    ptb_data = np.load('data/ud2.3_nl_las_copy/ud23nllas_biaffine_roberta_trial2_subset10.npy', allow_pickle = True)
    ptb_data = ptb_data.item()
    n_train = len(ptb_data['train'])
    num_rels = -1
    mu = 0
    lambd = 0.01 / n_train

    for set_name in ptb_data:
        util.preprocess_data(ptb_data[set_name], num_rels = 1)
    
    train_data = ptb_data['train']
    eval_data = ptb_data['dev']
    test_data = ptb_data['test']
    print('#train : {}, #eval : {}, #test : {} '.format(len(train_data), len(eval_data), len(test_data)))
    
    print('train...')
    opt_theta, _ = advtree.adv_train(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd)
    # opt_theta, _ = advtree.adv_train_pqn(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd)
    # opt_theta, _ = advtree.adv_stochastic_train(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd, batch_size = min(200, n_train), adv_method = 'marginal', prob_inf = False, learning_rate = 0.1)
    # opt_theta, _ = advtree.adv_stochastic_train(train_data, eval_data, test_data, num_rels = 1, mu = mu, lambd = lambd, batch_size = min(200, n_train), adv_method = 'game', prob_inf = False, learning_rate = 0.1)
    print('eval...')
    print('test...')

    return 0


if __name__ == '__main__':

    # main()
    multi_exps()

