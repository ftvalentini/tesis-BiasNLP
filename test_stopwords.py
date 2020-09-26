import numpy as np
import pandas as pd
import os, datetime

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.stats as stats
import nltk
from nltk.corpus import stopwords

from scripts.utils.corpora import load_vocab
from metrics.cooc import log_oddsratio

#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C3-V1.txt" # enwiki = C3
FILE_ODDSRATIO = "results/pkl/oddsratio_byword_MALE_SHORT-FEMALE_SHORT.csv"
FILE_GARG = "results/pkl/garg_byword_MALE_SHORT-FEMALE_SHORT.csv"

#%% Load data
print("Loading data...\n")
str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)
dat_odds = pd.read_csv(FILE_ODDSRATIO)
dat_garg = pd.read_csv(FILE_GARG)
dat = pd.merge(dat_odds, dat_garg, how='inner', on=['word','idx'])

#%% Transform data
# ranking segun abs(rel_norm_distance)
rnd_abs = dat['rel_norm_distance'].abs()
dat['rank_rnd_abs'] = stats.rankdata(rnd_abs, "average")/len(rnd_abs)
# ranking segun abs(log_oddsratio) solo si p-value < 0.01
lor_abs = dat.loc[dat['pvalue'] < 0.01]['log_oddsratio'].abs()
rank_lor_abs = stats.rankdata(lor_abs, "average")/len(lor_abs)
dat.loc[dat['pvalue'] < 0.01, 'rank_lor_abs'] = rank_lor_abs
dat.loc[dat['pvalue'] >= 0.01, 'rank_lor_abs'] = 0.0

#%% Keep stopwords
# words to remove from SW list
words_rm = ['he',"he's", 'him','his','himself','she',"she's",'her','hers','herself']
sw = set(stopwords.words('english')) - set(words_rm)
dat['sw'] = np.where(dat['word'].isin(sw), 1, 0)
dat_sw = dat.loc[dat['sw'] == 1]

#%% Plots

def scatter_stopwords(data, x_var, y_var):
    """
    Scatter plot with stopwords and RND vs LOR in axes
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        x_var, y_var, c='freq', s=10, cmap='viridis',norm=matplotlib.colors.LogNorm()
        ,data=data)
    ax.axhline(0, ls='--', color='black', linewidth=0.5)
    ax.axvline(0, ls='--', color='black', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_title('freq.')
    return ax



### 1: plot in original scale
g = scatter_stopwords(dat_sw, x_var='log_oddsratio', y_var='rel_norm_distance')
# add means
logodds_avg = log_oddsratio(
    dat_sw['count_context_a'].sum(), dat_sw['count_notcontext_a'].sum()
    , dat_sw['count_context_b'].sum(), dat_sw['count_notcontext_b'].sum()
    , ci_level=0.95)[0]
rnd_avg = dat_sw['rel_norm_distance'].mean()
g.scatter(x=logodds_avg, y=rnd_avg, color='r', marker="x")
fig_1 = g.get_figure()
fig_1.savefig('results/plots/scatter_stopwords_logodds_rnd.png', dpi=400)

### 2: plot with rankings of words in abs value
g = scatter_stopwords(dat_sw, x_var='rank_lor_abs', y_var='rank_rnd_abs')
# add means
rank_lor_avg = dat_sw['rank_lor_abs'].mean()
rank_rnd_avg = dat_sw['rank_rnd_abs'].mean()
g.scatter(x=rank_lor_avg, y=rank_rnd_avg, color='r', marker="x")
fig_2 = g.get_figure()
fig_2.savefig('results/plots/scatter_stopwords_ranklogodds_rankrnd.png', dpi=400)
# Nota: en LOR el rank=0 si pvalor>0.01

#%% Otros plots

def scatter_plt(data, x_var, y_var):
    """
    Scatter plot with all words
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        x_var, y_var, c='freq', s=1, cmap='viridis',norm=matplotlib.colors.LogNorm()
        ,data=data)
    ax.set_xscale('log')
    ax.axhline(0, ls='--', color='black', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_title('freq.')
    return ax

# Freq and RND
g = scatter_plt(dat, x_var='freq', y_var='rel_norm_distance')
fig_ = g.get_figure()
fig_.savefig('results/plots/scatter_freq_rnd.png', dpi=400)
# Freq and LOGOODS
g = scatter_plt(dat, x_var='freq', y_var='log_oddsratio')
fig_ = g.get_figure()
fig_.savefig('results/plots/scatter_freq_logodds.png', dpi=400)

def scatter_freq_sw(data, x_var, y_var):
    """
    Scatter plot with color facet for SW
    """
    data_sw = data[data['sw'] == 1]
    data_resto = data[data['sw'] == 0]
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    plt.scatter(x_var, y_var, linewidth=0, c='navy', s=4, data=data_resto)
    plt.scatter(x_var, y_var, linewidth=0, c='orange', s=4, data=data_sw)
    ax.axhline(0, ls='--', color='black', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    return ax

# Freq and RND
g = scatter_freq_sw(dat, x_var='freq', y_var='rel_norm_distance')
fig_ = g.get_figure()
fig_.savefig('results/plots/scatter_freq_rnd_sw.png', dpi=400)
# Freq and LOGODDS
g = scatter_freq_sw(dat, x_var='freq', y_var='log_oddsratio')
fig_ = g.get_figure()
fig_.savefig('results/plots/scatter_freq_logodds_sw.png', dpi=400)


#%% inspeccion visual
get_cols = ['word','freq','count_total','count_context_a','count_context_b' \
    ,'pmi_a','pmi_b','log_oddsratio', 'lower', 'upper', 'pvalue' \
    ,'rel_norm_distance', 'rank_rnd_abs', 'rank_lor_abs']
tmp = dat_sw[get_cols].copy()
tmp.loc[tmp['rank_rnd_abs'] > 0.9].sort_values('rank_rnd_abs', ascending=False)