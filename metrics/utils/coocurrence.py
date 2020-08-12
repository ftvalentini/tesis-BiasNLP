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
    return C.tocsr()


# document = """
# aa bb bb cc dd bb aa cc dd ee
# """
# word_list = ["aa", "bb", "dd"]
# str2idx = {"aa": 1, "bb": 2, "cc": 3, "dd": 4}
# window_size = 2
