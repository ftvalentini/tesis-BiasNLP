import numpy as np
import pandas as pd
import os, struct
from tqdm import tqdm


def bias_relative_norm_distance(embed_matrix
                                ,words_target_a, words_target_b, words_context
                                ,str2idx
                                ,ci_bootstrap_iters=None):
    """Return relative_norm_distance as bias metric between A/B wrt to Context
    (Garg et al 2018 Eq 3)
    Param:
        - embed_matrix: d x V matrix where column indices are indices of words
        given by str2idx
    """
    # handling words out of vocab
    words = words_target_a + words_target_b + words_context
    words_out = [w for w in words if w not in str2idx]
    if len(words_out) == len(words):
        raise ValueError("ALL WORDS ARE OUT OF VOCAB")
    if words_out:
        print(f'{", ".join(words_out)} \nNOT IN VOCAB')
    # words indices
    idx_a = sorted([str2idx[w] for w in words_target_a if w not in words_out])
    idx_b = sorted([str2idx[w] for w in words_target_b if w not in words_out])
    idx_c = sorted([str2idx[w] for w in words_context if w not in words_out])
    # normalize vecs to norm 1
        # Garg p8: "all vectors are normalized by their l2 norm"
    M = embed_matrix / np.linalg.norm(embed_matrix, axis=0)
    # average target vectors
    avg_a = np.mean(M[:,idx_a], axis=1)[:,np.newaxis]
    avg_b = np.mean(M[:,idx_b], axis=1)[:,np.newaxis]
    # matrix con context vectors (dim x n_words)
    M_c = M[:,idx_c]
    # para cada columna de C: substract avg_target y norma 2
    normdiffs_a = np.linalg.norm(np.subtract(M_c, avg_a), axis=0)
    normdiffs_b = np.linalg.norm(np.subtract(M_c, avg_b), axis=0)
    # relative norm distance: media de las diferencias de normdiffs target para cada C
    # rel_norm_distance > 0: mas lejos de B que de A --> bias "a favor de" A
    diffs = normdiffs_b - normdiffs_a
    rel_norm_distance = np.mean(diffs)
    # plots Garg usan sns.tsplot con error bands (deprecated function)
    if ci_bootstrap_iters:
        rel_norm_distances = []
        np.random.seed(123)
        for b in range(ci_bootstrap_iters):
            indices = np.random.choice(len(diffs), len(diffs), replace=True)
            rel_norm_distances.append(np.mean(diffs[indices]))
        sd = np.std(rel_norm_distances)
        lower, upper = rel_norm_distance - sd, rel_norm_distance + sd
        return rel_norm_distance, lower, upper
    return rel_norm_distance


def bias_byword(embed_matrix
                ,words_target_a, words_target_b, words_context
                ,str2idx, str2count):
    """ Return DataFrame with
        - Garg bias A/B of each Context word
        - freq of each word
    """
    # handling words out of vocab
    words = words_target_a + words_target_b + words_context
    words_out = [w for w in words if w not in str2idx]
    if len(words_out) == len(words):
        raise ValueError("ALL TARGET WORDS ARE OUT OF VOCAB")
    if words_out:
        print(f'{", ".join(words_out)} \nNOT IN VOCAB')
    # words inidices
    idx_a = sorted([str2idx[w] for w in words_target_a if w not in words_out])
    idx_b = sorted([str2idx[w] for w in words_target_b if w not in words_out])
    idx_c = sorted([str2idx[w] for w in words_context if w not in words_out])
    # normalize vecs to norm 1
        # Garg p8: "all vectors are normalized by their l2 norm"
    M = embed_matrix / np.linalg.norm(embed_matrix, axis=0)
    # average target vectors
    avg_a = np.mean(M[:,idx_a], axis=1)[:,np.newaxis]
    avg_b = np.mean(M[:,idx_b], axis=1)[:,np.newaxis]
    # matrix con context vectors (dim x n_words)
    M_c = M[:,idx_c]
    # para cada columna de C: substract avg_target y norma 2
    normdiffs_a = np.linalg.norm(np.subtract(M_c, avg_a), axis=0)
    normdiffs_b = np.linalg.norm(np.subtract(M_c, avg_b), axis=0)
    # relative norm distance: media de las diferencias de normdiffs target para cada C
    # rel_norm_distance > 0: mas lejos de B que de A --> bias "a favor de" A
    diffs = normdiffs_b - normdiffs_a
    # results DataFrame (todos los resultados sorted by idx)
    str2idx_context = {w: str2idx[w] for w in words_context if w not in words_out}
    str2count_context = {w: str2count[w] for w in str2idx_context}
    results = pd.DataFrame(str2idx_context.items(), columns=['word','idx'])
    results['rel_norm_distance'] = diffs
    results['freq'] = str2count_context.values()
    return results
