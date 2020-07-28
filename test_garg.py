import os
from utils.embeddings import get_embeddings
from utils.metrics_glove import bias_relative_norm_distance

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
str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)
embeddings = get_embeddings(word_list, str2idx, idx2str, embed_file=EMBED_FILE)

#%% Bias with relative norm distance
bias_garg = bias_relative_norm_distance(
                embeddings, words_a, words_b, words_c, ci_bootstrap_iters=200)

#%% print results
with open(f'results/biasgarg_{TARGET_A}-{TARGET_B}-{CONTEXT}.md', "w") as f:
    print(
        f'with {VOCAB_FILE} :\n'
        ,'\n### Bias by relative norm distance (Garg et al 2018) \n'
        ,f'- bias({TARGET_A},{TARGET_B},{CONTEXT}) = {bias_garg[0]}\n'
        ,f'- CI SE-bootrstap = {bias_garg[1]} -- {bias_garg[2]}\n'
        ,file = f
    )
