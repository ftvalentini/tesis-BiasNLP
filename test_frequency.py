import numpy as np
import pandas as pd
import sys
import datetime
import re

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scripts.utils.corpora import load_vocab
from metrics.glove import bias_byword


#%% Corpus parameters
# default values
VOCAB_FILE = "embeddings/vocab-C3-V20.txt"
EMBED_FILE = "embeddings/glove-C3-V20-W8-D1-D100-R0.05-E150-S1.npy"
# cmd values
if len(sys.argv)>1 and sys.argv[1].endswith(".txt"):
    VOCAB_FILE = sys.argv[1]
    EMBED_FILE = sys.argv[2]

res_id = re.search("^.+/(.+C\d+)-.+$", EMBED_FILE).group(1)

#%% target pairs parameters
TARGETS = [
    (["he"], ["she"])
    ,(["his","him"], ["her","hers"])
    ,(["himself"], ["herself"])
]

#%% Load dicts and WE
str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)
embed_matrix = np.load(EMBED_FILE)

#%% words lists
words_a, words_b = zip(*TARGETS)
words_a = [i for sublist in words_a for i in sublist]
words_b = [i for sublist in words_b for i in sublist]
words_context = [w for w in str2count.keys() if w not in words_a + words_b]

#%% WE bias by word
results = list()
for targets in TARGETS:
    df_bias = bias_byword(
        embed_matrix, targets[0], targets[1], words_context, str2idx, str2count)
    results.append(df_bias)

#%% Correlation between pairs

### check que dframes estan ordenados
assert np.all(results[0]['idx'] == results[1]['idx'])
###
gdat = results[0][['freq']]
for i, targets in enumerate(TARGETS):
    colname = f'rnd_{"-".join(targets[0])}_{"-".join(targets[1])}'
    gdat.loc[:,colname] = results[i].loc[:,'rel_cosine_similarity']

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
fig.savefig(f'results/plots/scatter_pronounpairs_corrs_{res_id}.png', dpi=400)


#%% Plots

fig, axs = plt.subplots(2, 2, figsize=(8,8), sharex='col', sharey=False)
axs = axs.ravel()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(len(results)):
    axs[i].scatter(
        'freq', 'rel_cosine_similarity', c='freq', cmap='viridis'
        ,s=1, norm=matplotlib.colors.LogNorm(), data=results[i])
    axs[i].set_xscale('log')
    axs[i].axhline(0, ls='--', color='black', linewidth=0.5)
    axs[i].set_xlabel('freq')
    axs[i].set_ylabel('rel_cosine_similarity')
    count1 = sum([str2count[w] for w in TARGETS[i][0]])
    count2 = sum([str2count[w] for w in TARGETS[i][1]])
    freq_ratio = np.round(count1 / count2, 2)
    name1 = "-".join(TARGETS[i][0])
    name2 = "-".join(TARGETS[i][1])
    axs[i].set_title(f'{name1}_{name2} (freq. ratio = {freq_ratio})')
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.set_title('freq.')
fig.savefig(f'results/plots/scatter_pronounpairs_freq_rnd_{res_id}.png', dpi=400)
