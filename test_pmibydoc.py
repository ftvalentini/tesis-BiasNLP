import os

from utils.metrics import bias_ppmi_bydoc

#%% Corpus parameters
CORPUS = 'corpora/simplewikiselect.txt' # wikipedia dump

#%% Estereotipos parameters
TARGET_A = 'MALE'
TARGET_B = 'FEMALE'
CONTEXT = 'SCIENCE'

#%% Read all Estereotipos
words_lists = dict()
for f in os.listdir('words_lists'):
    nombre = os.path.splitext(f)[0]
    words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]

#%% Get "diffential PMI" by doc
words_a = words_lists[TARGET_A]
words_b = words_lists[TARGET_B]
words_c = words_lists[CONTEXT]
result = bias_ppmi_bydoc(words_a, words_b, words_c, alpha=.75, window_size=8 \
                        ,corpus=CORPUS)

#%% MOST biased
most_biased = result.sort_values(by="diff_ppmi", ascending=False).head(5)
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
least_biased = result.sort_values(by="diff_ppmi", ascending=True).head(5)
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
with open("results/pmibydoc.md", "w") as f:
    print(
        f'with {CORPUS} :\n'
        ,f'\n### Most biased {TARGET_A}/{TARGET_B}/{CONTEXT} \n'
        ,most_biased
        ,f'\n### Most unbiased {TARGET_A}/{TARGET_B}/{CONTEXT} \n'
        ,least_biased
        ,file = f
    )
