import numpy as np
import pandas as pd
import os, datetime
import scipy.sparse
import scipy.stats as stats
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt

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
dat_sw = dat.loc[dat['word'].isin(sw)]

#%% Plots

def scatter_stopwords(data, x_var, y_var):
    """
    Scatter plot with stopwords and RND vs LOR in axes
    """
    g = sns.scatterplot(
        data=data, x=x_var, y=y_var
        , size="freq", size_norm=(plt_dat.freq.min(), plt_dat.freq.max())
        # , hue="freq" , hue_norm=(plt_dat.freq.min(), plt_dat.freq.max())
        # , marker="."
        )
    # ax = g.axes
    g.axvline(0, ls='--', color='black', linewidth=0.5)
    g.axhline(0, ls='--', color='black', linewidth=0.5)
    return g


### 1: plot in original scale
g = scatter_stopwords(dat_sw, x_var='log_oddsratio', y_var='rel_norm_distance')
# add means
lor_avg = log_oddsratio(
    dat_sw['count_context_a'].sum(), dat_sw['count_notcontext_a'].sum()
    , dat_sw['count_context_b'].sum(), dat_sw['count_notcontext_b'].sum()
    , ci_level=0.95)[0]
rnd_avg = dat_sw['rel_norm_distance'].mean()
g.scatter(x=lor_avg, y=rnd_avg, color='r', marker="x")
fig_1 = g.get_figure()
fig_1.savefig('results/plots/scatter_stopwords_lor_rnd.png', dpi=400)

### 2: plot with rankings of words in abs value
g = scatter_stopwords(dat_sw, x_var='rank_lor_abs', y_var='rank_rnd_abs')
# add means
rank_lor_avg = dat_sw['rank_lor_abs'].mean()
rank_rnd_avg = dat_sw['rank_rnd_abs'].mean()
g.scatter(x=rank_lor_avg, y=rank_rnd_avg, color='r', marker="x")
fig_2 = g.get_figure()
fig_2.savefig('results/plots/scatter_stopwords_ranklor_rankrnd.png', dpi=400)
# Nota: en LOR el rank=0 si pvalor>0.01

#%% inspeccion visual
get_cols = ['word','freq','count_total','count_context_a','count_context_b' \
    ,'pmi_a','pmi_b','log_oddsratio', 'lower', 'upper', 'pvalue' \
    , 'rel_norm_distance', 'rank_rnd_abs', 'rank_lor_abs']
tmp = dat_sw[get_cols].copy()
tmp.loc[tmp['rank_rnd_abs'] > 0.9].sort_values('rank_rnd_abs', ascending=False)


#%%OTROS

# max(list(str2idx.values())) # vocab size

## 20 mas sesgadas
# dat_join.loc[(dat_join['pvalue'] < 0.01)].sort_values(['log_oddsratio'], ascending=False).head(20)
# dat_join.loc[(dat_join['pvalue'] < 0.01)].sort_values(['log_oddsratio'], ascending=True).head(20)
# dat_join.sort_values(['rel_norm_distance'], ascending=False).head(20)
# dat_join.sort_values(['rel_norm_distance'], ascending=True).head(20)

# cols to get
get_cols = ['word','freq','count_total','count_context_a','count_context_b' \
    ,'pmi_a','pmi_b','log_oddsratio', 'lower', 'upper', 'pvalue' \
    , 'rel_norm_distance', 'rank_rnd_abs', 'rank_lor_abs']


# Palabras con menor/mayor diferencia en rank
tmp = dat_join[get_cols].copy()
tmp['dif_rank'] = abs(tmp['rank_rnd_abs'] - tmp['rank_lor_abs'])
tmp.sort_values(['dif_rank'], ascending=False).head(50)
tmp.loc[
    (dat_join['rank_rnd_abs'] > 0.9) & (dat_join['rank_lor_abs'] > 0.9)
].sort_values(['dif_rank'], ascending=True).head(50)

# Palabras con mayor diferencia en rank NO EN VALOR ABSOLUTO
tmp = dat_join[get_cols].copy()
tmp['rank_rnd'] = stats.rankdata(
    tmp['rel_norm_distance'], "average")/len(tmp['rel_norm_distance'])
lor = tmp.loc[tmp['pvalue'] < 0.01]['log_oddsratio']
rank_lor = stats.rankdata(lor, "average")/len(lor)
tmp.loc[tmp['pvalue'] < 0.01, 'rank_lor'] = rank_lor
tmp['dif_rank'] = abs(tmp['rank_rnd'] - tmp['rank_lor'])
tmp.loc[
    ((tmp['rank_rnd'] > 0.90) & (tmp['rank_lor'] < 0.10)) |
        ((tmp['rank_rnd'] < 0.10) & (tmp['rank_lor'] > 0.90))
].sort_values(['dif_rank'], ascending=False).head(50)


import seaborn as sns
sns.distplot(dat_join['log_oddsratio'])
sns.distplot(dat_join['rel_norm_distance'])
# stats.rankdata([0,1,100], "average")/len([0,1,100])


# 1. MALE cooc - FEMALE glove
# sorted by glove
tmp = dat_join.loc[
    (dat_join['upper'] > 0) & (dat_join['lower'] > 0) &
        (dat_join['pvalue'] < 0.01) &
        (dat_join['rel_norm_distance'] < 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by cooc
tmp = dat_join.loc[
    (dat_join['upper'] > 0) & (dat_join['lower'] > 0) &
        (dat_join['pvalue'] < 0.01) &
        (dat_join['rel_norm_distance'] < 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]

# 2. FEMALE cooc - MALE glove
# sorted by glove
tmp = dat_join.loc[
    (dat_join['upper'] < 0) & (dat_join['lower'] < 0) &
    (dat_join['pvalue'] < 0.01) &
    (dat_join['rel_norm_distance'] > 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by cooc
tmp = dat_join.loc[
    (dat_join['upper'] < 0) & (dat_join['lower'] < 0) &
    (dat_join['pvalue'] < 0.01) &
    (dat_join['rel_norm_distance'] > 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]

# 3. MALE cooc - MALE glove
# sorted by glove
tmp = dat_join.loc[
    (dat_join['upper'] > 0) & (dat_join['lower'] > 0) &
        (dat_join['pvalue'] < 0.01) &
        (dat_join['rel_norm_distance'] > 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by cooc
tmp = dat_join.loc[
    (dat_join['upper'] > 0) & (dat_join['lower'] > 0) &
        (dat_join['pvalue'] < 0.01) &
        (dat_join['rel_norm_distance'] > 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]

# 4. FEMALE cooc - FEMALE glove
# sorted by glove
tmp = dat_join.loc[
    (dat_join['upper'] < 0) & (dat_join['lower'] < 0) &
        (dat_join['pvalue'] < 0.01) &
        (dat_join['rel_norm_distance'] < 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by cooc
tmp = dat_join.loc[
    (dat_join['upper'] < 0) & (dat_join['lower'] < 0) &
        (dat_join['pvalue'] < 0.01) &
        (dat_join['rel_norm_distance'] < 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]
