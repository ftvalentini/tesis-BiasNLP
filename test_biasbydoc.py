import os

from utils.corpora import load_vocab
from utils.coocurrence import build_cooc_dict
from utils.metrics import differential_bias_bydoc

#%% Corpus parameters
CORPUS = 'corpora/simplewikiselect.txt' # wikipedia dump
CORPUS_METADATA = 'corpora/simplewikiselect.meta.json' # wikipedia dump
VOCAB_FILE = "embeddings/vocab-C0-V20.txt" # wikipedia dump = C0

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

#%% Get "diffential bias" by doc
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_c = words_lists[CONTEXT]
word_list = words_a + words_b + words_c
cooc_dict = build_cooc_dict(word_list, str2idx)
result = differential_bias_bydoc(
                words_a, words_b, words_c, cooc_dict, str2count
                ,metric="pmi", alpha=1.0, window_size=8
                ,corpus=CORPUS, corpus_metadata=CORPUS_METADATA)


#%% MOST biased
most_biased = result.sort_values(by="diff_bias", ascending=False).head(5)
indices = most_biased['line'].sort_values().to_list()
i = 0
docs_most_biased = []
with open(CORPUS, "r", encoding="utf-8") as f:
    while True:
        doc = f.readline().strip()
        if i in indices:
            docs_most_biased.append(doc)
        if not doc:
            break
        i += 1

#%% LEAST biased
least_biased = result.sort_values(by="diff_bias", ascending=True).head(5)
indices = least_biased['line'].sort_values().to_list()
i = 0
docs_least_biased = []
with open(CORPUS, "r", encoding="utf-8") as f:
    while True:
        doc = f.readline().strip()
        if i in indices:
            docs_least_biased.append(doc)
        if not doc:
            break
        i += 1

#%% print results
with open(f'results/diffbias_bydoc_{TARGET_A}-{TARGET_B}-{CONTEXT}.md', "w") as f:
    print(
        f'with {CORPUS} :\n'
        ,f'\n### Most biased {TARGET_A}/{TARGET_B}/{CONTEXT} \n'
        ,most_biased
        ,f'\n### Most unbiased {TARGET_A}/{TARGET_B}/{CONTEXT} \n'
        ,least_biased
        ,file = f
    )
