import numpy as np
import pandas as pd
import os, struct
from tqdm import tqdm

def norm_distances(embed_matrix, idx_target, idx_context):
    """Return relative norm distance values between avg of words_target and
    words_context
    Param:
        - embed_matrix: d x V matrix where column indices are indices of words
        given by str2idx
    Notes:
        - Must pass words indices (not words)
        - It works OK if len(idx_target) == 1
    """
    M = embed_matrix
    avg_target = np.mean(M[:,idx_target], axis=1)[:,np.newaxis]
    M_c = M[:,idx_context] # matriz de contexts words
    # distancia euclidiana
    normdiffs = np.linalg.norm(np.subtract(M_c, avg_target), axis=0)
    return normdiffs


def relative_norm_distances(embed_matrix, idx_target_a, idx_target_b
                            ,idx_context, str2idx):
    """Return relative norm distances between A/B wrt to each Context
    (Garg et al 2018 Eq 3)
    Param:
        - embed_matrix: d x V matrix where column indices are indices of words
        given by str2idx
    Notes:
        - Must pass words indices (not words)
    """
    # normalize vecs to norm 1
        # Garg p8: "all vectors are normalized by their l2 norm"
    M = embed_matrix / np.linalg.norm(embed_matrix, axis=0)
    # distancias de avg(target) cra cada context
    normdiffs_a = norm_distances(M, idx_target_a, idx_context)
    normdiffs_b = norm_distances(M, idx_target_b, idx_context)
    # rel_norm_distance > 0: mas lejos de B que de A --> bias "a favor de" A
    diffs = normdiffs_b - normdiffs_a
    return diffs


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
    # get differences dist(avg(B),c) - dist(avg(A),c) for each c
    diffs = relative_norm_distances(embed_matrix, idx_a, idx_b, idx_c, str2idx)
    # avg para todos los contextos
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


def bias_byword(embed_matrix, words_target_a, words_target_b, words_context
                ,str2idx, str2count):
    """ Return DataFrame with
        - Garg bias A/B of each Context word
        - freq of each word
    """
    # handling words out of vocab (asume que todas las context estan en vocab)
    words_target = words_target_a + words_target_b
    words_target_out = [w for w in words_target if w not in str2idx]
    if len(words_target_out) == len(words_target):
        raise ValueError("ALL TARGET WORDS ARE OUT OF VOCAB")
    if words_target_out:
        print(f'{", ".join(words_target_out)} \nNOT IN VOCAB')
    # words indices
    idx_a = sorted([str2idx[w] for w in words_target_a \
                                                if w not in words_target_out])
    idx_b = sorted([str2idx[w] for w in words_target_b \
                                                if w not in words_target_out])
    idx_c = sorted([str2idx[w] for w in words_context])
    # get differences dist(avg(B),c) - dist(avg(A),c) for each c
    diffs = relative_norm_distances(embed_matrix, idx_a, idx_b, idx_c, str2idx)
    # results DataFrame (todos los resultados sorted by idx)
    str2idx_context = {w: str2idx[w] for w in words_context}
    str2count_context = {w: str2count[w] for w in str2idx_context}
    results = pd.DataFrame(str2idx_context.items(), columns=['word','idx'])
    results['rel_norm_distance'] = diffs
    results['freq'] = str2count_context.values()
    return results
