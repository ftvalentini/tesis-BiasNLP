import numpy as np
import pandas as pd
import os, datetime

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# import scipy.sparse
# import scipy.stats as stats
# import nltk
# from nltk.corpus import stopwords

from scripts.utils.corpora import load_vocab
from metrics.glove import bias_byword

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C3-V20.txt" # enwiki = C3
EMBED_FILE = "embeddings/vectors-C3-V20-W8-D1-D100-R0.05-E150-S1.npy" # enwiki = C3

#%% target pairs parameters
TARGETS = [
    ("he", "she")
    ,("his", "hers")
    ,("him", "her")
    ,("himself", "herself")
]

#%% Load dicts and WE
str2idx, idx2str, str2count = load_vocab(vocab_file=VOCAB_FILE)
embed_matrix = np.load(EMBED_FILE)

#%% words lists
words_a = [w[0] for w in TARGETS]
words_b = [w[1] for w in TARGETS]
words_context = [w for w in str2count.keys() if w not in words_a + words_b]

#%% WE bias by word
results = list()
for words in zip(words_a, words_b):
    df_bias = bias_byword(
        embed_matrix, [words[0]], [words[1]], words_context, str2idx, str2count)
    results.append(df_bias)


#%% Correlation between pairs

gdat = results[0][['freq']]
for i, words in enumerate(zip(words_a, words_b)):
    colname = f'rnd_{words[0]}_{words[1]}'
    gdat.loc[:,colname] = results[i].loc[:,'rel_norm_distance']
### check que dframes estan ordenados
# np.all(results[0]['idx'] == results[1]['idx'])
###

features = [c for c in gdat.columns if c not in ['idx','word','freq']]
n_features = len(features)
fig, axs = plt.subplots(n_features, n_features, figsize=(10,10)
                        ,sharex=False, sharey=False)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
for i, f1 in enumerate(features):
    for j, f2 in enumerate(features):
        if i == j:
            axs[i][j].hist(gdat[f2], 50, density=True, alpha=0.3)
            axs[i][j].axvline(0, ls='--', color='black', linewidth=0.5)
            axs[i][j].set_xlabel(f2)
        if i > j:
            axs[i][j].scatter(f1, f2, c='freq', s=0.5, cmap='viridis'
                        ,norm=matplotlib.colors.LogNorm(), data=gdat)
            axs[i][j].axhline(0, ls='--', color='black', linewidth=0.5)
            axs[i][j].axvline(0, ls='--', color='black', linewidth=0.5)
            axs[i][j].set_ylabel(f1)
            axs[i][j].set_xlabel(f2)
fig.savefig('results/plots/scatter_pronounpairs_corrs.png', dpi=400)


#%% Plots

fig, axs = plt.subplots(2, 2, figsize=(8,8), sharex='col', sharey=False)
axs = axs.ravel()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(len(results)):
    axs[i].scatter(
        'freq', 'rel_norm_distance', c='freq', cmap='viridis'
        ,s=1, norm=matplotlib.colors.LogNorm(), data=results[i])
    axs[i].set_xscale('log')
    axs[i].axhline(0, ls='--', color='black', linewidth=0.5)
    axs[i].set_xlabel('freq')
    axs[i].set_ylabel('rel_norm_distance')
    w1 = words_a[i]
    w2 = words_b[i]
    freq_ratio = np.round(str2count[w1] / str2count[w2], 2)
    axs[i].set_title(f'{w1}-{w2} (freq. ratio = {freq_ratio})')
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.set_title('freq.')
fig.savefig('results/plots/scatter_pronounpairs_freq_rnd.png', dpi=400)
