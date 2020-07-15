import numpy as np
import pandas as pd
from ctypes import Structure, c_int, c_double, sizeof

# TEST: reproduce Cooc Glove
# (usando https://github.com/mebrunet/understanding-bias/blob/master/src/GloVe.jl#L172)
text = """
april is the fourth month of the year and comes between march and may it is one
of four months to have days april always begins on the same day of week as july
and additionally january in leap years april always ends on the same day of the
week as december
"""
VOCAB_MIN_COUNT = 4
WINDOW_SIZE = 1

words = [w.lower() for w in text.split()]
vocab = dict()
for w in words:
    vocab[w] = vocab.get(w, 0) + 1
vocab = {k:v for k,v in vocab.items() if v >= VOCAB_MIN_COUNT}

### Glove cooc
    # julia indexing empieza en 1
i = -1  # row index
j = -1  # col index
l1 = 1-1  # position of center word (resto 1 por indexing)
l2 = 0-1  # position of context word (resto 1 por indexing)
net_offset = 0  # offset (l1 - l2), excluding out-of-vocab "gaps"
rdo = dict()
# slide the center position
for l1, word in enumerate(words):
    # i = vocab.get(word, (-1,-1))[0]
    # if i == -1:
    i = word
    if i not in vocab:
        continue  # skip position if out-of-vocab
    l2 = l1  # align
    net_offset = 0  # reset
    while net_offset < WINDOW_SIZE:
        l2 -= 1  # increment
        if l2 <= 0:
            break
        # j = vocab.get(words[l2], (-1,-1))[0]
        # if j == -1:
        j = words[l2]
        if j not in vocab:
            continue
        net_offset += 1  # word in-vocab, so increment net offset
        rdo[(i,j)] = rdo.get((i,j),0) + 1
        rdo[(j,i)] = rdo.get((j,i),0) + 1
        print(i,l1,j,l2)
        # push!(I, i);
        # push!(J, j);
        # push!(vals, 1.0/net_offset)
### utils Cooc
document = text
word_list = vocab.keys()
cooc_dict = {}
words_doc = document.split()
for i, word_i in enumerate(words_doc):
    if word_i in word_list:
        window_start = max(0, i - WINDOW_SIZE)
        window_end = min(i + WINDOW_SIZE + 1, len(words_doc))
        for j in range(window_start, window_end):
            if (words_doc[j] in word_list) & (i != j):
                cooc_dict[(word_i, words_doc[j])] = \
                    cooc_dict.get((word_i, words_doc[j]), 0) + 1
                # cooc_dict.setdefault((word_i, words_doc[j]), 0)
                # cooc_dict[(word_i, words_doc[j])] += 1

rdo
cooc_dict




# TEST: cuanto da la suma coocs de .bin vs str2count
word = 'him'
idx = str2idx[word]
with open('embeddings/cooc-C0-V20-W8-D0.bin', 'rb') as f:
    result = []
    class CREC(Structure):
        _fields_ = [('word1', c_int), ('word2', c_int), ('value', c_double)]
    cr = CREC()
    while f.readinto(cr) == sizeof(CREC):
        if (cr.word1 == idx):
            result.append((cr.word1, cr.word2, cr.value))
        if cr.word1 > idx:
            break

sum([res[2] for res in result])
sum([res[2] for res in result if res[1] != idx])
str2count[word]
# --> no coinciden --> tengo que leer bin para hacer tabla de ORatio (creo)



# out of vocab:
# 'ingeniousness','ingeniously','brilliance','brilliantly','cleverness','intelligently'
# beautician, caregiver, dietician, florist, receptionist, typist



# def vector_similarity(vector_dict, word1, word2, tipo="cosine"):
#     """ Return cosine or euclidian distance between 2 vectors from a dict \
#     of vectors
#     """
#     if (word1 not in vector_dict) or (word2 not in vector_dict):
#         return float("nan")
#     # normalize vectors
#     v1 = vector_dict[word1] / np.linalg.norm(vector_dict[word1])
#     v2 = vector_dict[word2] / np.linalg.norm(vector_dict[word1])
#     if tipo == "cosine":
#         dist = np.dot(v1, v2)
#         # /np.linalg.norm(v1)/np.linalg.norm(v2)
#     else:
#         dist = np.linalg.norm(np.subtract(v1, v2))
#     return dist






text = """
One is the loneliest number that you will ever do
Two can be as bad as one
It is the loneliest number since the number one
No is the saddest experience you will ever know
Yes  it is the saddest experience you will ever know
Because one is the loneliest number that you will ever do
One is the loneliest number that you will ever know
It is just no good anymore since you went away
Now I spend my time just making rhymes of yesterday
Because one is the loneliest number that you will ever do
One is the loneliest number that you will ever know
One is the loneliest number
One is the loneliest number
One is the loneliest number that you will ever do
One is the loneliest number much much worse than two
One is the number divided by two
One
One is the loneliest number
"""
