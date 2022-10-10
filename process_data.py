import numpy as np
import os


deprel_dict = {'punct': 0, 'prep': 1, 'pobj': 2, 'det': 3, 'nn': 4, 'nsubj': 5, 'amod': 6, 'root': 7, 'dobj': 8, 'aux': 9, 'advmod': 10, 'conj': 11, 'cc': 12, 'num': 13, 'poss': 14, 'ccomp': 15, 'dep': 16, 'xcomp': 17, 'mark': 18, 'cop': 19, 'number': 20, 'possessive': 21, 'rcmod': 22, 'auxpass': 23, 'appos': 24, 'nsubjpass': 25, 'advcl': 26, 'partmod': 27, 'pcomp': 28, 'neg': 29, 'tmod': 30, 'quantmod': 31, 'npadvmod': 32, 'prt': 33, 'infmod': 34, 'parataxis': 35, 'mwe': 36, 'expl': 37, 'acomp': 38, 'iobj': 39, 'csubj': 40, 'predet': 41, 'preconj': 42, 'discourse': 43, 'csubjpass': 44}


def permute_ptb_data():

    overall_data = np.load('data/ptb/ptb_data_clean.npy', allow_pickle = True)
    overall_data = overall_data.item()
    new_data = {}
    for set_name in overall_data:
        new_data[set_name] = []
        for idx, instance in enumerate(overall_data[set_name]):
            if np.random.rand() < 0.1:
                new_data[set_name].append(instance)
    np.save('data/ptb/ptb_data_small_clean.npy', new_data)

    return 0


def get_random_tree(n):

    root = np.random.randint(n)
    rest = np.random.permutation(np.concatenate((np.arange(0, root), np.arange(root + 1, n))))
    nodes = np.concatenate(([root], rest))
    ret = np.zeros(n, dtype = int)
    ret[nodes[0]] = 0
    for i in range(1, n):
        ret[nodes[i]] = nodes[np.random.randint(i)] + 1

    return ret


def edit_tree(arcs, n_moves):

    n = arcs.size
    updated_mask = np.zeros(n, dtype = bool)
    for i in range(n_moves):
        u = np.random.randint(1, n + 1)
        par_u = np.random.choice(np.concatenate((np.arange(0, u), np.arange(u + 1, n + 1))))
        ori_par_u = arcs[u - 1]
        arcs[u - 1] = par_u
        updated_mask[u - 1] = True
        x = par_u
        while x != 0 and x != u:
            x = arcs[x - 1]
        if x == u:
            arcs[par_u - 1] = ori_par_u
            updated_mask[par_u - 1] = True

    return arcs, updated_mask


def perturb():

    input_path = 'data/ptb_sd330/train.conllx'
    output_path = 'data/ptb_sd330/train.noise80.conllx'
    fin = open(input_path, 'r')
    fout = open(output_path, 'w')
    p_noise = 0.8
    rels = list(deprel_dict.keys())
    num_rels = len(rels)
    
    cur_sentence = []
    for line in fin.readlines():
        tokens = line.strip().split()
        if len(tokens) == 10:
            cur_sentence.append(tokens)
        else:
            n = len(cur_sentence)
            dep_arcs = np.array([int(cur_sentence[i][6]) for i in range(n)], dtype = int)
            if np.random.rand() < p_noise:
                random_dep_arcs = get_random_tree(n)
                for i in range(n):
                    cur_sentence[i][6] = str(random_dep_arcs[i])
                    if random_dep_arcs[i] == 0:
                        cur_sentence[i][7] = 'root'
                    else:
                        cur_sentence[i][7] = np.random.choice(rels)
            
            '''
            n_moves = np.ceil(n * p_noise).astype(int)
            new_dep_arcs, rels_change_mask = edit_tree(dep_arcs, n_moves)
            for i in range(n):
                cur_sentence[i][6] = str(new_dep_arcs[i])
                if rels_change_mask[i]:
                    cur_sentence[i][7] = np.random.choice(rels)
            '''

            for tokens in cur_sentence:
                fout.write('\t'.join(tokens) + '\n')
            fout.write('\n')
            cur_sentence = []

    fin.close()
    fout.close()
    
    return 0


def write_data(data, file_path):

    with open(file_path, 'w') as fout:
        for sentence in data:
            for line in sentence:
                fout.write(line)

    return


def split_train():

    n_sep_train = 0
    base_folder = 'data/ud2.3_tur'
    input_path = os.path.join(base_folder, 'train.conllu')
    fin = open(input_path, 'r')
    all_data = []
    cur_sentence = []
    for line in fin.readlines():
        if line.startswith('#'):
            continue
        cur_sentence.append(line)
        if len(line.strip()) == 0:
            all_data.append(cur_sentence)
            cur_sentence = []
    n_data = len(all_data)
    print('# of data = {}'.format(n_data))
    all_data = np.array(all_data, dtype = object)
    
    fin.close()


    np.random.shuffle(all_data)
    sep_train_data = all_data[:n_sep_train]
    all_data = all_data[n_sep_train:]
    write_data(sep_train_data, os.path.join(base_folder, 'train.sep.conllu'))

    for trial_idx in range(5):
        for n_samples in [10, 50, 100, 1000]:
            np.random.shuffle(all_data)
            write_data(all_data[:n_samples], os.path.join(base_folder, 'train.trial{}.subset{}.conllu'.format(trial_idx, n_samples)))
    
    return 0


if __name__ == '__main__':

    split_train()

