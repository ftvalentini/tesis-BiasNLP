import os, datetime
import numpy as np
import pandas as pd
import scipy.sparse

from scripts.utils.corpora import load_vocab
from metrics.cooc import pmi, bias_odds_ratio

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C3-V20.txt" # wikipedia dump = C0
COOC_FILE = 'embeddings/cooc-C3-V1-W8-D0.npz'

#%% Estereotipos parameters
TARGET_A = 'CHRISTIANITY'
TARGET_B = 'ISLAM'
CONTEXT = 'ANGRY'

print("START:", datetime.datetime.now())

#%% Read all Estereotipos
words_lists = dict()
for f in os.listdir('words_lists'):
    nombre = os.path.splitext(f)[0]
    words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]

#%% Load corpus vocab dicts and cooc matrix
print("Loading data...\n")
str2idx, idx2str, str2count = load_vocab(vocab_file=VOCAB_FILE)
cooc_matrix = scipy.sparse.load_npz(COOC_FILE)

#%% TEST: pmi(A,c)-pmi(B,c) near bias_odds_ratio(A,B,c)
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_c = words_lists[CONTEXT]
pmi_a = pmi(cooc_matrix, words_a, words_c, str2idx, alpha=1)
pmi_b = pmi(cooc_matrix, words_b, words_c, str2idx, alpha=1)
odds_ratio = bias_odds_ratio(
    cooc_matrix, words_a, words_b, words_c, str2idx, ci_level=.95)
# logOddsRatio = 0 --> no bias
# logOddsRatio > 0 --> bias A
# logOddsRatio < 0 --> bias B

# #%% check raw counts
# count_a = sum([str2count[word] for word in words_a])
# count_b = sum([str2count[word] for word in words_b])
# count_a_c = sum([cooc_dict.get((word, context), 0) \
#                             for word in words_a for context in words_c])
# count_b_c = sum([cooc_dict.get((word, context), 0) \
#                             for word in words_b for context in words_c])
# # coocurrences A - C
# df_ac = pd.DataFrame(
#             [(word, context, cooc_dict.get((word,context),0)) \
#                                 for word in words_a for context in words_c])
# # coocurrences B - C
# df_bc = pd.DataFrame(
#             [(word, context, cooc_dict.get((word,context),0)) \
#                                 for word in words_b for context in words_c])
# # df_ac.sort_values(by=2, ascending=False)
# # df_bc.sort_values(by=2, ascending=False)

#%% print results
with open(f'results/oddsratio_{TARGET_A}-{TARGET_B}-{CONTEXT}.md', "w") as f:
    print(
        f'with {VOCAB_FILE} :\n'
        ,'\n### PMI and Odds Ratio \n'
        ,f'- pmi({TARGET_A},{CONTEXT}) = {pmi_a}\n'
        ,f'- pmi({TARGET_B},{CONTEXT}) = {pmi_b}\n'
        ,f'- diff_pmi = {pmi_a - pmi_b}\n'
        ,f'- logOddsRatio = {np.log(odds_ratio[0])}\n'
        ,f'- CI-logOddsRatio = {np.log(odds_ratio[1])} -- {np.log(odds_ratio[2])}\n'
        # ,'\n### Word counts \n'
        # ,f'- {TARGET_A}: {count_a}\n'
        # ,f'- {TARGET_B}: {count_b}\n'
        # ,f'- {TARGET_A} & {CONTEXT}: {count_a_c}\n'
        # ,f'- {TARGET_B} & {CONTEXT}: {count_b_c}\n'
        ,file = f
    )

print("END:", datetime.datetime.now())
