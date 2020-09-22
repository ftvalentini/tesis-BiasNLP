


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
