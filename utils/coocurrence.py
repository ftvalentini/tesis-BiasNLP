from ctypes import Structure, c_int, c_double, sizeof


class CREC(Structure):
    """c++ class to read triples (idx, idx, cooc) from binary file
    """
    _fields_ = [('idx1', c_int),
                ('idx2', c_int),
                ('value', c_double)]


def build_cooc_dict(word_list, str2idx,
                    cooc_file='embeddings/cooc-C0-V20-W8-D0.bin'):
    """
    Build coocurrence matrix for word_list from data in binary glove file \
    and str2idx built with glove vocab
    Order of column/rows is order of list
    """
    words_outof_vocab = [word for word in word_list if word not in str2idx]
    if words_outof_vocab:
        raise KeyError(f'{", ".join(words_outof_vocab)} \nNOT IN VOCAB')
    # size of data in bytes
    size_crec = sizeof(CREC)
    # init matrix or dict
    # cooc_mat = np.full((len(word_list), len(word_list)), 0)
    cooc_mat = dict()
    # indices to insert data in matrix
    # idx2row = {str2idx[j]: i for i, j in enumerate(word_list)}
    idx2str = {str2idx[word]: word for word in word_list}
    # words indices sorted from most frequent to less
    indices = sorted([str2idx[word] for word in word_list])
    # open bin file
    with open(cooc_file, 'rb') as f:
        # binary file is sorted by idx1
        # --> we update current_idx each time we finished search of idx1
        cr = CREC()
        k = 0
        current_idx = indices[0]
        # read and overwrite into cr while there is data
        while (f.readinto(cr) == size_crec):
            if cr.idx1 in indices:
                if cr.idx2 in indices:
                    # cooc_mat[idx2row[cr.idx1], idx2row[cr.idx2]] = cr.value
                    cooc_mat[(idx2str[cr.idx1], idx2str[cr.idx2])] = cr.value
                # if new idx1 is higher than previous and is target word --> update
                if cr.idx1 > current_idx:
                    k += 1
                    current_idx = indices[k]
            # stop if idx1 is higher than highest search idx
            if cr.idx1 > indices[len(indices) - 1]:
                break
    return cooc_mat


def create_cooc_dict(document, word_list, window_size=8):
    """ Create coocurrence dictionary (sparse matrix) from a document for word_list
    """
    # sabemos que simplewikiselect no tiene puntuacion
    # --> ver gensim.corpora.wikicorpus.WikiCorpus.get_texts()
    cooc_dict = {}
    words_doc = document.split()
    for i, word_i in enumerate(words_doc):
        if word_i in word_list:
            window_start = max(0, i - window_size)
            window_end = min(i + window_size + 1, len(words_doc))
            for j in range(window_start, window_end):
                if (words_doc[j] in word_list) & (i != j):
                    cooc_dict.setdefault((word_i, words_doc[j]), 0)
                    cooc_dict[(word_i, words_doc[j])] += 1
    return cooc_dict


def find_cooc(word1, word2, str2idx, cooc_file='embeddings/cooc-C0-V20-W8-D0.bin'):
    """
    Find one cooc value from binary glove file and str2idx built with glove vocab
    """
    if word1 not in str2idx:
        raise KeyError(f'{word1} not in vocab')
    if word2 not in str2idx:
        raise KeyError(f'{word2} not in vocab')
    size_crec = sizeof(CREC)
    idx = (str2idx[word1], str2idx[word2])
    i = min(idx)
    j = max(idx)
    with open(cooc_file, 'rb') as f:
        result = []
        cr = CREC()
        while f.readinto(cr) == size_crec:
            if (cr.word1 == i) & (cr.word2 == j):
                break
        result.append((cr.word1, cr.word2, cr.value))
    return result[0][2]
