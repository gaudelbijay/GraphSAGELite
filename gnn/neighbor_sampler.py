import numpy as np


def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):
    _sample = np.random.choice  # speed hack (local pointer to the function)
    neighs = [list(G[int(node)]) for node in nodes]
    if sample_num:
        if self_loop:
            sample_num = -1
        samp_neighs = [
            list(_sample(neigh, sample_num, replace=False)) if len(neigh) >= sample_num else list(
                _sample(neigh, sample_num, replace=True)) for neigh in neighs]
        if self_loop:
            samp_neighs = [samp_neighs + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
    else:
        samp_neighs = neighs
    return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))
