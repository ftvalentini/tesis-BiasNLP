import numpy as np
import pandas as pd
import datetime
import sys
import re

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.stats as stats
import nltk
from nltk.corpus import stopwords
from statsmodels.nonparametric.smoothers_lowess import lowess

from metrics.cooc import log_oddsratio


#%% Corpus parameters
# default values
FILE_RESULTS_DPMI = "results/csv/dpmibyword_C3_MALE_SHORT-FEMALE_SHORT.csv"
FILE_RESULTS_WE = "results/csv/distbyword_w2v-C3_MALE_SHORT-FEMALE_SHORT.csv"
# cmd values
if len(sys.argv)>1 and sys.argv[1].endswith(".csv"):
    FILE_RESULTS_DPMI = sys.argv[1]
    FILE_RESULTS_WE = sys.argv[2]

res_id = re.search("^.+/distbyword_(.+C\d+)_.+$", FILE_RESULTS_WE).group(1)

#%% Load data
print("Loading data...\n")
dat_dpmi = pd.read_csv(FILE_RESULTS_DPMI)
dat_we = pd.read_csv(FILE_RESULTS_WE)
dat = pd.merge(dat_dpmi, dat_we, how='inner', on=['word','idx'])

#%% Transform data
# ranking segun abs(rel_norm_distance)
rnd_abs = dat['rel_norm_distance'].abs()
dat['rank_rnd_abs'] = stats.rankdata(rnd_abs, "average")/len(rnd_abs)
# ranking segun abs(rel_cosine_similarity)
rcs_abs = dat['rel_cosine_similarity'].abs()
dat['rank_rcs_abs'] = stats.rankdata(rcs_abs, "average")/len(rcs_abs)
# ranking segun abs(log_oddsratio) solo si p-value < 0.01
dpmi_abs = dat.loc[dat['pvalue'] < 0.01]['diff_pmi'].abs()
rank_dpmi_abs = stats.rankdata(dpmi_abs, "average")/len(dpmi_abs)
dat.loc[dat['pvalue'] < 0.01, 'rank_dpmi_abs'] = rank_dpmi_abs
dat.loc[dat['pvalue'] >= 0.01, 'rank_dpmi_abs'] = 0.0

#%% Keep stopwords
# words to remove from SW list
words_rm = ['he',"he's", 'him','his','himself','she',"she's",'her','hers','herself']
sw = set(stopwords.words('english')) - set(words_rm)
dat['sw'] = np.where(dat['word'].isin(sw), 1, 0)
dat_sw = dat.loc[dat['sw'] == 1]

#%% Plot fns

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


def scatter_plt(data, x_var, y_var, n_sample=None, smooth=False, frac=0.1, seed=123):
    """
    Scatter plot with all words (x with log10)
    Param:
        - n_sample: size of sample or None
        - lowess: plot smooth curve or not
    """
    if n_sample:
        data = data.sample(n_sample, random_state=seed)
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        x_var, y_var, c='freq', s=0.7, cmap='viridis',norm=matplotlib.colors.LogNorm()
        ,data=data)
    if smooth:
        smooth_data = lowess(data[y_var], np.log10(data[x_var]), frac=frac)
        line = ax.plot(
            10**smooth_data[:,0], smooth_data[:,1], color='red', lw=1.0, ls='--')
    ax.set_xscale('log')
    ax.axhline(0, ls='--', color='black', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_title('freq.')
    return ax


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


#%% 1: plot in original scale
g = scatter_stopwords(dat_sw, x_var='diff_pmi', y_var='rel_cosine_similarity')
# add means
### nota: no se puede calcular el DPMI exacto porque hace falta la suma
### de los counts de toda la matriz de cooc
### --> uso odds_ratio para el promedio
logodds_avg = log_oddsratio(
    dat_sw['count_context_a'].sum(), dat_sw['count_notcontext_a'].sum()
    , dat_sw['count_context_b'].sum(), dat_sw['count_notcontext_b'].sum()
    , ci_level=0.95)[0]
rcs_avg = dat_sw['rel_cosine_similarity'].mean()
g.scatter(x=logodds_avg, y=rcs_avg, color='r', marker="x")
fig_1 = g.get_figure()
fig_1.savefig(f'results/plots/scatter_stopwords_dpmi_rcs_{res_id}.png', dpi=400)


#%% 2: plot with rankings of words in abs value
g = scatter_stopwords(dat_sw, x_var='rank_dpmi_abs', y_var='rank_rcs_abs')
# add means
rank_lor_avg = dat_sw['rank_dpmi_abs'].mean()
rank_rnd_avg = dat_sw['rank_rcs_abs'].mean()
g.scatter(x=rank_lor_avg, y=rank_rnd_avg, color='r', marker="x")
fig_2 = g.get_figure()
fig_2.savefig(f'results/plots/scatter_stopwords_rankdpmi_rankrcs_{res_id}.png', dpi=400)
# Nota: en LOR el rank=0 si pvalor>0.01


# #%% RND vs RCS
# g = scatter_stopwords(dat, x_var='rel_norm_distance', y_var='rel_cosine_similarity')
# fig_ = g.get_figure()
# fig_.savefig(f'results/plots/scatter_rnd_rcs_{res_id}.png', dpi=400)


# #%% Freq and RND
# g = scatter_plt(dat, x_var='freq', y_var='rel_norm_distance')
# fig_ = g.get_figure()
# fig_.savefig(f'results/plots/scatter_freq_rnd_{res_id}.png', dpi=400)


#%% Freq and RCS
g = scatter_plt(dat, x_var='freq', y_var='rel_cosine_similarity')
fig_ = g.get_figure()
fig_.savefig(f'results/plots/scatter_freq_rcs_{res_id}.png', dpi=400)


#%% Freq and RCS (smooth)
g = scatter_plt(
    dat, x_var='freq', y_var='rel_cosine_similarity', n_sample=20_000, smooth=True, frac=0.1)
fig_ = g.get_figure()
fig_.savefig(f'results/plots/scatter_freq_rcs_smooth_{res_id}.png', dpi=400)


#%% Freq and RCS (smooth, only SW)
g = scatter_plt(
    dat_sw, x_var='freq', y_var='rel_cosine_similarity', smooth=True, frac=0.5)
fig_ = g.get_figure()
fig_.savefig(f'results/plots/scatter_freq_rcs_sw_smooth_{res_id}.png', dpi=400)


#%% Freq and DPMI
g = scatter_plt(dat, x_var='freq', y_var='diff_pmi')
fig_ = g.get_figure()
fig_.savefig(f'results/plots/scatter_freq_dpmi_{res_id}.png', dpi=400)


# #%% stopwords: Freq and RND
# g = scatter_freq_sw(dat, x_var='freq', y_var='rel_norm_distance')
# fig_ = g.get_figure()
# fig_.savefig(f'results/plots/scatter_freq_rnd_sw_{res_id}.png', dpi=400)


#%% stopwords: Freq and RCS
g = scatter_freq_sw(dat, x_var='freq', y_var='rel_cosine_similarity')
fig_ = g.get_figure()
fig_.savefig(f'results/plots/scatter_freq_rcs_sw_{res_id}.png', dpi=400)


#%% stopwords: Freq and DPMI
g = scatter_freq_sw(dat, x_var='freq', y_var='diff_pmi')
fig_ = g.get_figure()
fig_.savefig(f'results/plots/scatter_freq_dpmi_sw_{res_id}.png', dpi=400)


#%% inspeccion visual
# get_cols = ['word','freq','count_total','count_context_a','count_context_b' \
#     ,'pmi_a','pmi_b','log_oddsratio', 'lower', 'upper', 'pvalue' \
#     ,'rel_norm_distance', 'rank_rnd_abs', 'rank_lor_abs']
# tmp = dat_sw[get_cols].copy()
# tmp.loc[tmp['rank_rnd_abs'] > 0.9].sort_values('rank_rnd_abs', ascending=False)
