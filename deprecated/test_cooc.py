import os
import numpy as np
import pandas as pd

from utils.corpora import load_vocab
from utils.coocurrence import build_cooc_dict, create_cooc_dict

#%% Corpus parameters (must run scripts/01-cooc.sh scripts/cooc.config)
CORPUS = "corpora/test_short.txt"
CORPUS_IDX = 2 # segun 01-cooc.sh
WINDOW_SIZE = 1
VOCAB_MIN_COUNT = 4

#%% Load raw corpus
with open(CORPUS, "r", encoding="utf-8") as f:
    corpus = f.readlines()
all_words = list()
for linea in corpus:
    for word in linea.split():
        all_words.append(word)

#%% Load corpus vocab dicts
VOCAB_FILE = f'embeddings/vocab-C{CORPUS_IDX}-V{VOCAB_MIN_COUNT}.txt'
str2idx, idx2str, str2count = load_vocab(vocab_file=VOCAB_FILE)

#%% unique words
word_counts = pd.Series(all_words).value_counts()
word_counts = word_counts[word_counts >= VOCAB_MIN_COUNT]
word_list = word_counts.index.to_list()
word_list_glove = list(str2idx.keys())

#%% word counts
different_count = list()
for word in word_list_glove:
    count_glove = str2count.get(word, 0)
    count_utils = word_counts[word]
    if count_glove != count_utils:
        different_count.append(f'{word}: glove: {count_glove} -- utils: {count_utils}')

#%% cooc counts
COOC_FILE = f'embeddings/cooc-C{CORPUS_IDX}-V{VOCAB_MIN_COUNT}-W{WINDOW_SIZE}-D0.bin'
cooc_dict = create_cooc_dict(" ".join(all_words), word_list_glove, window_size=WINDOW_SIZE)
cooc_dict_glove = build_cooc_dict(word_list_glove, str2idx, cooc_file=COOC_FILE)
different_cooc = list()
for word1 in word_list:
    for word2 in word_list:
        k = (word1, word2)
        cooc_glove = cooc_dict_glove.get(k, 0)
        cooc_utils = cooc_dict.get(k, 0)
        if cooc_glove != cooc_utils:
            different_cooc.append(f'{k}: glove: {cooc_glove} -- utils: {cooc_utils}')

#%% print results
with open("results/comparecooc.md", "w") as f:
    print(
        f'with {VOCAB_FILE} and {COOC_FILE}'
        ,'\n### Word counts'
        ,f'Glove: {len(word_list_glove)} -- utils {len(word_list)}'
        ,f'words in common: {len(set(word_list) & set(word_list_glove))}'
        ,'\n### Word counts with difference'
        ,*different_count
        ,'\n### Coocs. with difference'
        ,*different_cooc
        ,sep='\n'
        ,file = f
    )
