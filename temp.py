import numpy as np
import pandas as pd
from ctypes import Structure, c_int, c_double, sizeof


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
