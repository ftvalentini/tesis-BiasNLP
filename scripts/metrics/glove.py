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


def cosine_similarities(embed_matrix, idx_target, idx_context, use_norm=True):
    """Return relative cosin similarity values between avg of words_target and
    words_context
    Param:
        - embed_matrix: d x V matrix where column indices are indices of words
        given by str2idx
        - use_norm: divides by norm (as usual cosine) -- if False: only dot pr.
    Notes:
        - Must pass words indices (not words)
        - It works OK if len(idx_target) == 1
    """
    M = embed_matrix
    avg_target = np.mean(M[:,idx_target], axis=1)[:,np.newaxis]
    M_c = M[:,idx_context] # matriz de contexts words
    # similitud coseno (dot product)
    # (context van a estar normalizados -- targets se "desnormalizan" al promediar)
    # rel_sims = np.dot(avg_target.T, M_c)
    productos = np.dot(avg_target.T, M_c)
    normas = np.linalg.norm(M_c, axis=0)
    denominadores = np.linalg.norm(avg_target) * normas
    out = productos.ravel()
    if use_norm:
        out /= denominadores.ravel()
    return out


def relative_norm_distances(
                        embed_matrix, idx_target_a, idx_target_b, idx_context):
    """Return relative norm distances between A/B wrt to each Context
    (Garg et al 2018 Eq 3)
    Param:
        - embed_matrix: d x V matrix where column indices are indices of words
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


def relative_cosine_diffs(
    embed_matrix, idx_target_a, idx_target_b, idx_context, use_norm=True
    , return_both=False):
    """Return relative norm distances between A/B wrt to each Context
    (Garg et al 2018 Eq 3)
    Param:
        - embed_matrix: d x V matrix where column indices are indices of words
        - return_both: returns tuple (cos(a,c), cos(b,c))
        - use_norm: divides by norm (as usual cosine) -- if False: only dot pr.
    Notes:
        - Must pass words indices (not words)
    """
    # normalize vecs to norm 1
        # Garg p8: "all vectors are normalized by their l2 norm"
    # --> NOT DONE (cosine ya normaliza)
    # distancias de avg(target) cra cada context
    cos_a = cosine_similarities(
                    embed_matrix, idx_target_a, idx_context, use_norm=use_norm)
    cos_b = cosine_similarities(
                    embed_matrix, idx_target_b, idx_context, use_norm=use_norm)
    # > 0: mas cerca de A que de B --> bias "hacia" A
    if return_both:
        return cos_a, cos_b
    diffs = cos_a - cos_b
    return diffs


def bias_embeddings(embed_matrix
                    ,words_target_a, words_target_b, words_context
                    ,str2idx
                    ,type="norm2"
                    ,ci_bootstrap_iters=None):
    """Return bias metric between A/B wrt to Context
    (Garg et al 2018 Eq 3)
    Param:
        - embed_matrix: d x V matrix where column indices are indices of words
        given by str2idx
        - type: "norm2" or "cosine"
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
    # get differences for each c
    if type == "norm2":
        diffs = relative_norm_distances(
                                    embed_matrix, idx_a, idx_b, idx_c)
    if type == "cosine":
        diffs = relative_cosine_similarities(
                                    embed_matrix, idx_a, idx_b, idx_c)
    # avg para todos los contextos
    mean_diff = np.mean(diffs)
    # plots Garg usan sns.tsplot con error bands (deprecated function)
    if ci_bootstrap_iters:
        means = []
        np.random.seed(123)
        for b in range(ci_bootstrap_iters):
            indices = np.random.choice(len(diffs), len(diffs), replace=True)
            means.append(np.mean(diffs[indices]))
        sd = np.std(means)
        lower, upper = mean_diff - sd, mean_diff + sd
        return mean_diff, lower, upper
    return mean_diff


def bias_byword(embed_matrix, words_target_a, words_target_b, words_context
                ,str2idx, str2count):
    """ Return DataFrame with
        - Embedding bias A/B of each Context word
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
    # get differences for each c
    diffs_norm2 = relative_norm_distances(embed_matrix, idx_a, idx_b, idx_c)
    dots_a, dots_b = relative_cosine_diffs(
            embed_matrix, idx_a, idx_b, idx_c, use_norm=False, return_both=True)
    cosines_a, cosines_b = relative_cosine_diffs(
            embed_matrix, idx_a, idx_b, idx_c, use_norm=True, return_both=True)
    # normas
    M_c = embed_matrix[:,idx_c]
    normas = np.linalg.norm(M_c, axis=0)
    # results DataFrame (todos los resultados sorted by idx)
    str2idx_context = {w: str2idx[w] for w in words_context}
    str2count_context = {w: str2count[w] for w in str2idx_context}
    results = pd.DataFrame(str2idx_context.items(), columns=['word','idx'])
    results['rel_norm_distance'] = diffs_norm2
    results['freq'] = str2count_context.values()
    results['cosine_a'] = cosines_a
    results['cosine_b'] = cosines_b
    results['dot_a'] = dots_a
    results['dot_b'] = dots_b
    results['diff_cosine'] = cosines_a - cosines_b
    results['diff_dot'] = dots_a - dots_b
    results['norm'] = normas
    return results
