import numpy as np
import pandas as pd
import os, struct, datetime

from scripts.utils.corpora import load_vocab
from metrics.glove import bias_byword

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C3-V20.txt" # wikipedia dump = C0
EMBED_FILE = "embeddings/vectors-C3-V20-W8-D1-D100-R0.05-E150-S1.npy" # wikipedia dump = C0

#%% Estereotipos parameters
TARGET_A = 'MALE_SHORT'
TARGET_B = 'FEMALE_SHORT'

print("START:", datetime.datetime.now())

#%% Read all Estereotipos
words_lists = dict()
for f in os.listdir('words_lists'):
    nombre = os.path.splitext(f)[0]
    words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]

#%% Load corpus vocab dicts
str2idx, idx2str, str2count = load_vocab(vocab_file=VOCAB_FILE)
embed_matrix = np.load(EMBED_FILE)

#%% words lists
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_context = [w for w in str2count.keys() if w not in words_a + words_b]

#%% relative norm difference by word
print("Computing bias wrt each context word...")
result = bias_byword(
    embed_matrix, words_a, words_b, words_context, str2idx, str2count)
most_biased = result.\
                sort_values(['rel_norm_distance','freq'], ascending=[False, False]). \
                head(20)
least_biased = result.\
                sort_values(['rel_norm_distance','freq'], ascending=[True, False]). \
                head(20)

#%% save csv results
result.to_csv(f'results/csv/rnd_byword_{TARGET_A}-{TARGET_B}.csv', index=False)

#%% print results
with open(f'results/rnd_byword_{TARGET_A}-{TARGET_B}.md', "w") as f:
    print(
        f'with {EMBED_FILE} :\n'
        ,f'\n### Most biased {TARGET_A}/{TARGET_B} \n'
        ,most_biased.to_markdown()
        ,f'\n### Most unbiased {TARGET_A}/{TARGET_B} \n'
        ,least_biased.to_markdown()
        ,file = f
    )

print("END:", datetime.datetime.now())
