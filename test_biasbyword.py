import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.corpora import load_vocab
from utils.coocurrence import build_cooc_dict
from utils.metrics import pmi, bias_odds_ratio, bias_byword

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C0-V20.txt" # wikipedia dump = C0
COOC_FILE = 'embeddings/cooc-C0-V20-W8-D0.bin' # wikipedia dump = C0

#%% Estereotipos parameters
TARGET_A = 'MALE'
TARGET_B = 'FEMALE'

#%% Read all Estereotipos
words_lists = dict()
for f in os.listdir('words_lists'):
    nombre = os.path.splitext(f)[0]
    words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]

#%% Load corpus vocab dicts
str2idx, idx2str, str2count = load_vocab(vocab_file=VOCAB_FILE)

#%% TEST: Odds Ratio A/B for every word in vocab
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_context = [w for w in str2count.keys() if w not in words_a + words_b]
rdos = bias_byword(words_a, words_b, words_context, str2idx, str2count
                , cooc_file=COOC_FILE)
rdos['log_odds_ratio'] = np.log(rdos['odds_ratio'])
most_biased = rdos.\
                loc[np.isfinite(rdos['pvalue'])]. \
                sort_values(['odds_ratio','pvalue'], ascending=[False, True]). \
                head(10)
least_biased = rdos.\
                loc[np.isfinite(rdos['pvalue'])]. \
                sort_values(['odds_ratio','pvalue'], ascending=[True, True]). \
                head(10)


#%% print results
with open(f'results/oddsratio_byword_{TARGET_A}-{TARGET_B}.md', "w") as f:
    print(
        f'with {COOC_FILE} :\n'
        ,f'\n### Most biased {TARGET_A}/{TARGET_B} \n'
        ,most_biased
        ,f'\n### Most unbiased {TARGET_A}/{TARGET_B} \n'
        ,least_biased
        ,file = f
    )
