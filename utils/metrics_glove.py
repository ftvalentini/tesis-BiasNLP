import numpy as np
import pandas as pd



def bias_relative_norm_distance(vector_dict
                                ,words_target_a, words_target_b, words_context
                                ,ci_bootstrap_iters=None):
    """Return relative_norm_distance as bias metric between A/B wrt to Context
    (Garg et al 2018 Eq 3)
    """
    # TODO: handlear words out of vocab
    # normalize vecs to norm 1
    for word, vec in vector_dict.items():
        vector_dict[word] = vec / np.linalg.norm(vec)
    # average target vectors
    avg_a = np.mean(np.array([vector_dict[w] for w in words_target_a]), axis=0)
    avg_b = np.mean(np.array([vector_dict[w] for w in words_target_b]), axis=0)
    # array context vectors (dim x n_words)
    vectors_c = np.array([vector_dict[w] for w in words_context]).transpose()
    # para cada columna de C: substract avg_target y norma 2
    normdiffs_a = np.linalg.norm(np.subtract(vectors_c, avg_a[:,np.newaxis]), axis=0)
    normdiffs_b = np.linalg.norm(np.subtract(vectors_c, avg_b[:,np.newaxis]), axis=0)
    # relative norm distance: suma de las diferencias de normdiffs target para cada C
    # rel_norm_distance > 0: mas lejos de B que de A --> bias "a favor de" A
    rel_norm_distance = np.sum(normdiffs_b - normdiffs_a)
    if ci_bootstrap_iters:
        diffs = normdiffs_b - normdiffs_a
        rel_norm_distances = []
        np.random.seed(123)
        for b in range(ci_bootstrap_iters):
            indices = np.random.choice(len(diffs), len(diffs), replace=True)
            rel_norm_distances.append(np.sum(diffs[indices]))
        sd = np.std(rel_norm_distances)
        lower, upper = rel_norm_distance - sd, rel_norm_distance + sd
        return rel_norm_distance, lower, upper
    return rel_norm_distance
