import os, datetime
import numpy as np

from scripts.utils.corpora import load_vocab
from metrics.glove import bias_relative_norm_distance

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C3-V20.txt" # wikipedia dump = C0
EMBED_FILE = "embeddings/vectors-C3-V20-W8-D1-D100-R0.05-E150-S1.npy" # wikipedia dump = C0

#%% Estereotipos parameters
TARGET_A = 'ECUADOR'
TARGET_B = 'EUROPE'
CONTEXT = 'COCAINE'

print("START:", datetime.datetime.now())

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
str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)
embed_matrix = np.load(EMBED_FILE)

#%% Bias with relative norm distance
bias_garg = bias_relative_norm_distance(
                            embed_matrix, words_a, words_b, words_c, str2idx
                            ,ci_bootstrap_iters=200)

#%% print results
with open(f'results/biasgarg_{TARGET_A}-{TARGET_B}-{CONTEXT}.md', "w") as f:
    print(
        f'with {VOCAB_FILE} :\n'
        ,'\n### Bias by relative norm distance (Garg et al 2018) \n'
        ,f'- bias({TARGET_A},{TARGET_B},{CONTEXT}) = {bias_garg[0]}\n'
        ,f'- CI SE-bootrstap = {bias_garg[1]} -- {bias_garg[2]}\n'
        ,file = f
    )

print("END:", datetime.datetime.now())
