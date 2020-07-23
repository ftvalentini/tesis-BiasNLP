import os
import numpy as np

from utils.corpora import load_vocab
from utils.coocurrence import build_cooc_dict
from utils.metrics import pmi, bias_odds_ratio

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C0-V20.txt" # wikipedia dump = C0
COOC_FILE = 'embeddings/cooc-C0-V20-W8-D0.bin'
WINDOW_SIZE = 8

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

#%% TEST: pmi(A,c)-pmi(B,c) near bias_odds_ratio(A,B,c)
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_c = words_lists[CONTEXT]
cooc_dict = build_cooc_dict(words_a + words_b, words_c, str2idx
                            ,cooc_file=COOC_FILE)
pmi_a = pmi(cooc_dict, words_a, words_c, str2count, WINDOW_SIZE, alpha=1)
pmi_b = pmi(cooc_dict, words_b, words_c, str2count, WINDOW_SIZE, alpha=1)

# NOTA: los pmi dan negativos porque la prod condicional calculada con los conteos \
# totales segun glove es muy baja
# TODO: ver que hacer!!!!

# window_size=8; alpha=1
# words_target = words_a; words_context = words_c
#     count_context_alpha = sum(
#         [str2count.get(w, 0)**alpha for w in words_context])
#     count_total_alpha = sum([v**alpha for v in str2count.values()])
#     prob_context_alpha = count_context_alpha / count_total_alpha
#     # probabilidad condicional de coocurrencia en la ventana
#     count_context_window = 0
#     for target in words_target:
#         for context in words_context:
#             count_context_window += cooc_dict.get((target, context), 0)
#     count_window_total = sum(
#         [str2count.get(w, 0)*window_size*2 for w in words_target])
#     prob_context_window = count_context_window / count_window_total
#     pmi = np.log(prob_context_window / prob_context_alpha)

odds_ratio = bias_odds_ratio(
    cooc_dict, words_a, words_b, words_c, str2count, ci_level=.95)
# logOddsRatio = 0 --> no bias
# logOddsRatio > 0 --> bias A
# logOddsRatio < 0 --> bias B

#%% check raw counts
count_a = sum([str2count[word] for word in words_a])
count_b = sum([str2count[word] for word in words_b])
count_a_c = sum([cooc_dict.get((word, context), 0) \
                            for word in words_a for context in words_c])
count_b_c = sum([cooc_dict.get((word, context), 0) \
                            for word in words_b for context in words_c])
# coocurrences A - C
[(word, context, cooc_dict.get((word,context),0)) for word in words_a for context in words_c]
# coocurrences B - C
[(word, context, cooc_dict.get((word,context),0)) for word in words_b for context in words_c]

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
        ,'\n### Word counts \n'
        ,f'- {TARGET_A}: {count_a}\n'
        ,f'- {TARGET_B}: {count_b}\n'
        ,f'- {TARGET_A} & {CONTEXT}: {count_a_c}\n'
        ,f'- {TARGET_B} & {CONTEXT}: {count_b_c}\n'
        ,file = f
    )
