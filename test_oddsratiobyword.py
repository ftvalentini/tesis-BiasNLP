import numpy as np
import pandas as pd
import os, datetime
import scipy.sparse

from scripts.utils.corpora import load_vocab
from metrics.cooc import pmi, bias_odds_ratio, bias_byword

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C3-V1.txt" # wikipedia dump = C0
COOC_FILE = "embeddings/cooc-C3-V1-W8-D0.npz"

#%% Estereotipos parameters
TARGET_A = 'MALE'
TARGET_B = 'FEMALE'
WORD_MIN_COUNT = 20

print("START:", datetime.datetime.now())

#%% Read all Estereotipos
words_lists = dict()
for f in os.listdir('words_lists'):
    nombre = os.path.splitext(f)[0]
    words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]

#%% Load corpus vocab dicts
print("Loading data...\n")
str2idx, idx2str, str2count = load_vocab(vocab_file=VOCAB_FILE)
cooc_matrix = scipy.sparse.load_npz(COOC_FILE)

#%% TEST: Odds Ratio A/B for every word in vocab
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_context = [w for w, freq in str2count.items() if \
                        w not in words_a + words_b and freq >= WORD_MIN_COUNT]
print("Computing results...\n")
rdos = bias_byword(cooc_matrix, words_a, words_b, words_context, str2idx)
# most_biased = rdos.\
#                 loc[np.isfinite(rdos['pvalue'])]. \
#                 sort_values(['odds_ratio','pvalue'], ascending=[False, True]). \
#                 head(20)
# least_biased = rdos.\
#                 loc[np.isfinite(rdos['pvalue'])]. \
#                 sort_values(['odds_ratio','pvalue'], ascending=[True, True]). \
#                 head(20)

#%% save pickle results
rdos.to_csv(f'results/pkl/oddsratio_byword_{TARGET_A}-{TARGET_B}.csv', index=False)

#%% print results
# with open(f'results/oddsratio_byword_{TARGET_A}-{TARGET_B}.md', "w") as f:
#     print(
#         f'with {COOC_FILE} :\n'
#         ,f'\n### Most biased {TARGET_A}/{TARGET_B} \n'
#         ,most_biased.to_markdown()
#         ,f'\n### Most unbiased {TARGET_A}/{TARGET_B} \n'
#         ,least_biased.to_markdown()
#         ,file = f
#     )

print("END:", datetime.datetime.now())
