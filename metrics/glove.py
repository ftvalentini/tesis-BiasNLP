import numpy as np
import pandas as pd
import os, struct
from tqdm import tqdm


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


def bias_byword(vector_a, vector_b, words_context, str2idx, str2count
                ,embed_file="embeddings/vectors-C0-V20-W8-D1-D50-R0.05-E100-S1.bin"):
    """ Return DataFrame with
        - Garg bias A/B of each Context word
        - freq of each word
    """
    vocab_size = len(str2idx)
    n_weights = os.path.getsize(embed_file) // 8 # 8 bytes per double
    embed_dim = (n_weights - 2*vocab_size) // (2*vocab_size)
        # dos vectores por word + 2 vectores constantes (biases)
        # see https://github.com/mebrunet/understanding-bias/blob/master/src/GloVe.jl
    # el bin esta ordenado por indice -- tienen vector dim 50 + 1 bias
    indices = sorted([str2idx[w] for w in words_context])
    idx2str = {str2idx[w]: w for w in words_context}
    str2bias = dict()
    # read embed_dim weights (double) + 1 bias (double) by word
    with open(embed_file, 'rb') as f:
        # idx starts in 1 in idx2str
        for i in tqdm(range(1, vocab_size+1)):
            embedding = struct.unpack('d'*embed_dim, f.read(8*embed_dim))
            bias = struct.unpack('d'*1, f.read(8*1)) # 'd' for double
            if i > indices[-1]:
                break
            if i in indices:
                embedding /= np.linalg.norm(embedding) # normaliza vector
                normdiff_a = np.linalg.norm(np.subtract(embedding, vector_a))
                normdiff_b = np.linalg.norm(np.subtract(embedding, vector_b))
                str2bias[idx2str[i]] = np.sum(normdiff_b - normdiff_a)
    str2freq = {k: str2count[k] for k in str2count.keys() if k in words_context}
    result = pd.DataFrame({'rel_norm_distance': str2bias, 'freq': str2freq})
    return result
