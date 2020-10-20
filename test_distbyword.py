import numpy as np
import pandas as pd
import os
import sys
import datetime
import re

from scripts.utils.corpora import load_vocab
from metrics.glove import bias_byword


#%% Parameters
# default values
VOCAB_FILE = "embeddings/vocab-C3-V20.txt"
EMBED_FILE = "embeddings/glove-C3-V20-W8-D1-D100-R0.05-E150-S1.npy"
TARGET_A = 'MALE_SHORT'
TARGET_B = 'FEMALE_SHORT'
# cmd values
if len(sys.argv)>1 and sys.argv[1].endswith(".npy"):
    VOCAB_FILE = sys.argv[1]
    EMBED_FILE = sys.argv[2]
    if len(sys.argv) >= 4:
        TARGET_A = sys.argv[3]
        TARGET_B = sys.argv[4]

print("START:", datetime.datetime.now())

#%% Read all Estereotipos
words_lists = dict()
for f in os.listdir('words_lists'):
    nombre = os.path.splitext(f)[0]
    words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]

#%% Load corpus vocab dicts
str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)
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
                sort_values(['rel_cosine_similarity','freq'], ascending=[False, False]). \
                head(20)
least_biased = result.\
                sort_values(['rel_cosine_similarity','freq'], ascending=[True, False]). \
                head(20)

#%% save csv results
results_name = re.search("^.+/(.+C\d+)-.+$", EMBED_FILE).group(1)
result.to_csv(
    f'results/csv/distbyword_{results_name}_{TARGET_A}-{TARGET_B}.csv', index=False)

#%% print results
with open(f'results/distbyword_{results_name}_{TARGET_A}-{TARGET_B}.md', "w" \
         , encoding="utf-8") as f:
    print(
        f'with {EMBED_FILE} :\n'
        ,f'\n### Most biased {TARGET_A}/{TARGET_B} \n'
        ,most_biased.to_markdown()
        ,f'\n### Most unbiased {TARGET_A}/{TARGET_B} \n'
        ,least_biased.to_markdown()
        ,file = f
    )

print("END:", datetime.datetime.now())
