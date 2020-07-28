import numpy as np
import pandas as pd
import os, struct

from utils.corpora import load_vocab
from utils.embeddings import get_embeddings
from utils.metrics_glove import bias_byword

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

#%% Load corpus vocab dicts
str2idx, idx2str, str2count = load_vocab(vocab_file=VOCAB_FILE)

#%% create embeddings of target words
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_context = [w for w in str2count.keys() if w not in words_a + words_b]
vectors_target = get_embeddings(words_a + words_b, str2idx, embed_file=EMBED_FILE)
# normaliza embeddings
for word, vec in vectors_target.items():
    vectors_target[word] = vec / np.linalg.norm(vec)
# embedding promedio de cada target
vector_avg_a = np.mean(np.array([vectors_target[w] for w in words_a]), axis=0)
vector_avg_b = np.mean(np.array([vectors_target[w] for w in words_b]), axis=0)

#%% relative norm difference by word
result = bias_byword(vector_avg_a, vector_avg_b, words_context, str2idx, str2count
                ,embed_file=EMBED_FILE)
most_biased = result.\
                sort_values(['rel_norm_distance','freq'], ascending=[False, False]). \
                head(20)
least_biased = result.\
                sort_values(['rel_norm_distance','freq'], ascending=[True, False]). \
                head(20)

#%% save pickle results
result.to_pickle(f'results/pkl/garg_byword_{TARGET_A}-{TARGET_B}.pkl')

#%% print results
with open(f'results/garg_byword_{TARGET_A}-{TARGET_B}.md', "w") as f:
    print(
        f'with {EMBED_FILE} :\n'
        ,f'\n### Most biased {TARGET_A}/{TARGET_B} \n'
        ,most_biased.to_markdown()
        ,f'\n### Most unbiased {TARGET_A}/{TARGET_B} \n'
        ,least_biased.to_markdown()
        ,file = f
    )
