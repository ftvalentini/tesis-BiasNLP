from ctypes import Structure, c_int, c_double, sizeof
from tqdm import tqdm


class CREC(Structure):
    """c++ class to read triples (idx, idx, cooc) from binary file
    """
    _fields_ = [('idx1', c_int),
                ('idx2', c_int),
                ('value', c_double)]


def build_cooc_dict(words_target, words_context, str2idx,
                    cooc_file='embeddings/cooc-C0-V20-W8-D0.bin'):
    """
    Build coocurrence matrix for words_target and words_context from data in \
    binary glove file and str2idx built with glove vocab
    Order of column/rows is order of list
    """
    words_outof_vocab = [word for word in words_target + words_context \
                                                        if word not in str2idx]
    if words_outof_vocab:
        print(f'{", ".join(words_outof_vocab)} \nNOT IN VOCAB')
    idx2str = {str2idx[word]: word for word in words_target + words_context}
    # words indices sorted from most frequent to less
    target_indices = sorted([str2idx[word] for word in words_target])
    context_indices = sorted([str2idx[word] for word in words_context])
    cooc_dict = dict() # init dict
    size_crec = sizeof(CREC) # crec: structura de coocucrrencia en Glove
    pbar = tqdm(total=target_indices[-1]*len(str2idx))
    # open bin file sorted by idx1
    with open(cooc_file, 'rb') as f:
        # --> we update current_idx each time we finish search of idx1
        cr = CREC()
        k = 0
        current_idx = target_indices[0]
        # read and overwrite into cr while there is data
        while (f.readinto(cr) == size_crec):
            pbar.update(1)
            if cr.idx1 in target_indices:
                if cr.idx2 in context_indices:
                    cooc_dict[(idx2str[cr.idx1], idx2str[cr.idx2])] = cr.value
                    cooc_dict[(idx2str[cr.idx2], idx2str[cr.idx1])] = cr.value
                # if new idx1 and is target is higher than previous --> update
                if cr.idx1 > current_idx:
                    k += 1
                    current_idx = target_indices[k]
            # stop if idx1 is higher than highest target idx
            if cr.idx1 > target_indices[-1]:
                break
    pbar.close()
    return cooc_dict


def create_cooc_dict(document, word_list, window_size=8):
    """ Create coocurrence dictionary (sparse matrix) from a document for word_list
    It assumes data has no punctuation
    """
    cooc_dict = {}
    words_doc = document.split()
    for i, word_i in enumerate(words_doc):
        if word_i in word_list:
            window_start = max(0, i - window_size)
            window_end = min(i + window_size + 1, len(words_doc))
            for j in range(window_start, window_end):
                if (words_doc[j] in word_list) & (i != j):
                    cooc_dict[(word_i, words_doc[j])] = \
                        cooc_dict.get((word_i, words_doc[j]), 0) + 1
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
