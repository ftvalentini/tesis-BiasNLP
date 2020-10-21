import numpy as np
import scipy.sparse


def create_cooc_matrix(document, word_list, str2idx, window_size=8):
    """ Create coocurrence sparse matrix from a document with a str2idx
    with non-zero values only for words in word_list
    Row-col indices are given by indices in str2idx
    It assumes data has no punctuation
    """
    V = max(str2idx.values())
    C = scipy.sparse.dok_matrix((V+1,V+1), dtype=np.float32)
    tokens = document.split()
    for i, word_i in enumerate(tokens):
        if word_i in word_list:
            window_start = max(0, i - window_size)
            window_end = min(i + window_size + 1, len(tokens))
            for j in range(window_start, window_end):
                if (tokens[j] in word_list) & (i != j):
                    C[str2idx[word_i], str2idx[tokens[j]]] += 1
    return C.tocoo().tocsr()


def create_cooc_matrix2(document, word_list, str2idx, window_size=8):
    """ Create coocurrence sparse matrix from a document with a str2idx
    with non-zero values only for words in word_list
    Row-col indices are given by indices in str2idx
    It assumes data has no punctuation
    Este algoritmo itera por palabra, registrando "dos veces" las coocurrencias
    de window_size palabras mas atras
    """
    V = max(str2idx.values())
    C = scipy.sparse.dok_matrix((V+1,V+1), dtype=np.float32)
    tokens = document.split()
    l1 = 0  # position of center word
    l2 = 0  # position of context word
    net_offset = 0  # offset (l1 - l2)
    for l1, token_i in enumerate(tokens):
        l2 = l1  # align
        net_offset = 0  # reset
        while net_offset < window_size:
            l2 -= 1
            if l2 < 0:
                break
            token_j = tokens[l2]
            net_offset += 1
            if (token_i in word_list) and (token_j in word_list):
                C[str2idx[token_i], str2idx[token_j]] += 1
                C[str2idx[token_j], str2idx[token_i]] += 1
    return C.tocoo().tocsr()


### TEST: matrix creation efficiency
# document = "aa bb bb cc dd bb aa cc dd ee" * 300
# vocab = list(set(document.split()))
# str2idx = {w: i+1 for i, w in enumerate(vocab)}
# word_list = list(str2idx.keys())
# word_list = ['aa','cc','dd']
# window_size = 8
# %timeit create_cooc_matrix(document, word_list, str2idx, window_size)
# %timeit create_cooc_matrix2(document, word_list, str2idx, window_size)
# aa = create_cooc_matrix(document, word_list, str2idx, window_size)
# bb = create_cooc_matrix2(document, word_list, str2idx, window_size)
# assert((aa.todense() == bb.todense()).all())
