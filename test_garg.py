import os
from utils.embeddings import get_embeddings
from utils.metrics import bias_relative_norm_distance

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C0-V20.txt" # wikipedia dump = C0
EMBED_FILE = "embeddings/vectors-C0-V20-W8-D1-D50-R0.05-E100-S1.bin" # wikipedia dump = C0

#%% Estereotipos parameters
TARGET_A = 'MALE'
TARGET_B = 'FEMALE'
CONTEXT = 'SCIENCE'

#%% Read all Estereotipos
words_lists = dict()
for f in os.listdir('words_lists'):
    nombre = os.path.splitext(f)[0]
    words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]

#%% Load embeddings
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_c = words_lists[CONTEXT]
word_list = words_a + words_b + words_c
embeddings = get_embeddings(word_list, vocab_file=VOCAB_FILE, embed_file=EMBED_FILE)

#%% Bias with relative norm distance
bias_garg = bias_relative_norm_distance(embeddings, words_a, words_b, words_c)

#%% print results
with open("results_biasgarg.md", "w") as f:
    print(
        f'with {VOCAB_FILE} :\n'
        ,'\n### Bias by relative norm distance (Garg et al 2018) \n'
        ,f'- bias({TARGET_A},{TARGET_B},{CONTEXT}) = {bias_garg}\n'
        ,file = f
    )
