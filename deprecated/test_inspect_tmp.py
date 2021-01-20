import numpy as np
import pandas as pd
import datetime
import sys
import re

import scipy.stats as stats


#%% Corpus parameters
# default values
FILE_RESULTS_DPMI = "results/csv/dpmibyword_C3_MALE_SHORT-FEMALE_SHORT.csv"
FILE_RESULTS_WE = "results/csv/distbyword_w2v-C3_MALE_SHORT-FEMALE_SHORT.csv"
# cmd values
if len(sys.argv)>1 and sys.argv[1].endswith(".csv"):
    FILE_RESULTS_DPMI = sys.argv[1]
    FILE_RESULTS_WE = sys.argv[2]

#%% Load data
dat_dpmi = pd.read_csv(FILE_RESULTS_DPMI)
dat_we = pd.read_csv(FILE_RESULTS_WE)
dat = pd.merge(dat_dpmi, dat_we, how='inner', on=['word','idx'])

#%% Transform data
# ranking segun abs(rel_norm_distance)
rnd_abs = dat['rel_norm_distance'].abs()
dat['rank_rnd_abs'] = stats.rankdata(rnd_abs, "average")/len(rnd_abs)
# ranking segun abs(rel_cosine_similarity)
rcs_abs = dat['rel_cosine_similarity'].abs()
dat['rank_rcd_abs'] = stats.rankdata(rcs_abs, "average")/len(rcs_abs)
# ranking segun abs(log_oddsratio) solo si p-value < 0.01
lor_abs = dat.loc[dat['pvalue'] < 0.01]['log_oddsratio'].abs()
rank_lor_abs = stats.rankdata(lor_abs, "average")/len(lor_abs)
dat.loc[dat['pvalue'] < 0.01, 'rank_lor_abs'] = rank_lor_abs
dat.loc[dat['pvalue'] >= 0.01, 'rank_lor_abs'] = 0.0

# cols to get
get_cols = ['word','freq','count_total','count_context_a','count_context_b' \
    ,'pmi_a','pmi_b','log_oddsratio', 'lower', 'upper', 'pvalue' \
    , 'rel_norm_distance', 'rank_rnd_abs', 'rank_lor_abs']

#%% Explora previo

### vocab size
# max(list(str2idx.values()))

# ### 20 mas sesgadas
# dat.sort_values(['rel_norm_distance'], ascending=False).head(50)[get_cols]
# dat.sort_values(['rel_norm_distance'], ascending=True).head(50)[get_cols]
# dat.loc[(dat['pvalue'] < 0.01)].sort_values(['log_oddsratio'], ascending=False).head(50)[get_cols]
# dat.loc[(dat['pvalue'] < 0.01)].sort_values(['log_oddsratio'], ascending=True).head(50)[get_cols]
#
# ### distribuciones
# import seaborn as sns
# sns.distplot(dat['log_oddsratio'])
# sns.distplot(dat['rel_norm_distance'])
# sns.distplot(dat.query("freq > 2000 & freq < 10_000")['rel_cosine_similarity'])
# # stats.rankdata([0,1,100], "average")/len([0,1,100])


#%% Explore For Drive Sheet

# 1. MALE cooc - FEMALE WE
# sorted by WE
tmp = dat.loc[
    (dat['upper'] > 0) & (dat['lower'] > 0) &
        (dat['pvalue'] < 0.01) &
        (dat['rel_norm_distance'] < 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by DPMI
tmp = dat.loc[
    (dat['upper'] > 0) & (dat['lower'] > 0) &
        (dat['pvalue'] < 0.01) &
        (dat['rel_norm_distance'] < 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]

# 2. FEMALE cooc - MALE glove
# sorted by glove
tmp = dat.loc[
    (dat['upper'] < 0) & (dat['lower'] < 0) &
    (dat['pvalue'] < 0.01) &
    (dat['rel_norm_distance'] > 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by cooc
tmp = dat.loc[
    (dat['upper'] < 0) & (dat['lower'] < 0) &
    (dat['pvalue'] < 0.01) &
    (dat['rel_norm_distance'] > 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]

# 3. MALE cooc - MALE glove
# sorted by glove
tmp = dat.loc[
    (dat['upper'] > 0) & (dat['lower'] > 0) &
        (dat['pvalue'] < 0.01) &
        (dat['rel_norm_distance'] > 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by cooc
tmp = dat.loc[
    (dat['upper'] > 0) & (dat['lower'] > 0) &
        (dat['pvalue'] < 0.01) &
        (dat['rel_norm_distance'] > 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]

# 4. FEMALE cooc - FEMALE glove
# sorted by glove
tmp = dat.loc[
    (dat['upper'] < 0) & (dat['lower'] < 0) &
        (dat['pvalue'] < 0.01) &
        (dat['rel_norm_distance'] < 0)
].sort_values(['rank_rnd_abs'], ascending=False).head(20)
tmp[get_cols]
# sorted by cooc
tmp = dat.loc[
    (dat['upper'] < 0) & (dat['lower'] < 0) &
        (dat['pvalue'] < 0.01) &
        (dat['rel_norm_distance'] < 0)
].sort_values(['rank_lor_abs'], ascending=False).head(20)
tmp[get_cols]


# Palabras con menor/mayor diferencia en rank
# (a)
tmp = dat[get_cols].copy()
tmp['dif_rank'] = abs(tmp['rank_rnd_abs'] - tmp['rank_lor_abs'])
tmp.sort_values(['dif_rank'], ascending=False).head(50)
# (b)
tmp.loc[
    (dat['rank_rnd_abs'] > 0.9) & (dat['rank_lor_abs'] > 0.9)
].sort_values(['dif_rank'], ascending=True).head(50)




# # Palabras con mayor diferencia en rank NO EN VALOR ABSOLUTO (??)
# tmp = dat[get_cols].copy()
# tmp['rank_rnd'] = stats.rankdata(
#     tmp['rel_norm_distance'], "average")/len(tmp['rel_norm_distance'])
# lor = tmp.loc[tmp['pvalue'] < 0.01]['log_oddsratio']
# rank_lor = stats.rankdata(lor, "average")/len(lor)
# tmp.loc[tmp['pvalue'] < 0.01, 'rank_lor'] = rank_lor
# tmp['dif_rank'] = abs(tmp['rank_rnd'] - tmp['rank_lor'])
# tmp.loc[
#     ((tmp['rank_rnd'] > 0.90) & (tmp['rank_lor'] < 0.10)) |
#         ((tmp['rank_rnd'] < 0.10) & (tmp['rank_lor'] > 0.90))
# ].sort_values(['dif_rank'], ascending=False).head(50)
