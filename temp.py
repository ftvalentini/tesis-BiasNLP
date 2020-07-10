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



# # OLD:
# get num_lines
# size_crec = 16
# f = open("embeddings/cooc-C0-V20-W8.bin", 'rb')
# f.seek(0, 2) # fseeko(fin, 0, SEEK_END)
# file_size = f.tell()
# num_lines = file_size / size_crec
# f.close()
# # read cooc
# num_threads = 8
# f = open("embeddings/cooc-C0-V20-W8.bin", 'rb')
# f.seek((num_lines / num_threads * id) * size_crec, 0)
# cr = f.read(size_crec, 1)
# f.close()
# steps
# 1. fopen("rb")
# 2. fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET);
# num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
# int num_threads = 8;
# id va de 0 a 8 (creo)
# sizeof(CREC):
# file_size = ftello(fin);
# 3. fread: fread(&cr, sizeof(CREC), 1, fin);


# lineas = list()
# with open("corpora/simplewikiselect.txt", "r") as f:
#     for i in range(10):
#         lineas.append(f.readline())
# lineas[:3]
