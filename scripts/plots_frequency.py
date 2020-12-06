import numpy as np
import pandas as pd
import datetime
import sys
import re

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess

CORPUS_ID = 3
TARGET_A = "HE"
TARGET_B = "SHE"

file_results = \
        f"../results/csv/biasbyword_full-C{CORPUS_ID}_{TARGET_A}-{TARGET_B}.csv"
        # f"results/csv/biasbyword_full-C{CORPUS_ID}_{TARGET_A}-{TARGET_B}.csv"


dat = pd.read_csv(file_results)


#%% Plots by metric

def scatter_plt(data, x_var, y_var, n_sample=None, smooth=False, frac=0.1, seed=123):
    """
    Scatter plot (x with log10)
    Param:
        - n_sample: size of sample or None
        - lowess: plot smooth curve or not
        - frac: lowess frac
    """
    if n_sample:
        data = data.sample(n_sample, random_state=seed)
    data_sw = data[data['sw'] == 1]
    data_resto = data[data['sw'] == 0]
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xlim(left=10, right=10**8)
    plt.scatter(x_var, y_var, linewidth=0, c='darkolivegreen', s=4, data=data_resto)
    plt.scatter(x_var, y_var, linewidth=0, c='firebrick', s=4, data=data_sw)
    if smooth:
        smooth_data = lowess(data[y_var], np.log10(data[x_var]), frac=frac)
        line = ax.plot(
            10**smooth_data[:,0], smooth_data[:,1], color='black', lw=1.0, ls='--')
    ax.axhline(0, ls='--', color='gray', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    return fig, ax


def boxplots_plt(data, x_var, y_var, bins):
    """
    Cut x_var in bins and make one boxplot per bin
    """
    freq_bins = pd.cut(np.log10(data[x_var]), bins=bins)
    nobs = freq_bins.value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    fig, ax = plt.subplots()
    ax = sns.boxplot(x=freq_bins, y=data[y_var], showfliers=False)
    ax.axhline(0, ls='--', color='black', linewidth=0.5)
    labels_ypos = ax.get_ylim()[1]
    for i in range(len(nobs)):
        ax.text(
            i, labels_ypos, nobs[i], horizontalalignment='center', size='small'
            , color='black', weight='semibold')
    return fig, ax


dat_sw = dat.loc[dat['sw'] == 1]
limits = [1,2,3,4,5,6,8]
metrics = ['dppmi', 'order2', 'cosine_glove', 'cosine_w2v']
for m in metrics:
    result_name = f"{m}-C{CORPUS_ID}_{TARGET_A}-{TARGET_B}"
    # freq-bias all words
    fig_, ax_ = scatter_plt(dat, x_var='freq', y_var=m)
    fig_.savefig(f'../results/plots/scatter_freq_{result_name}.png', dpi=400)
    # freq-bias sample+smooth
    fig_, ax_ = scatter_plt(
        dat, x_var='freq', y_var=m, n_sample=20_000, smooth=True, frac=0.075)
    fig_.savefig(f'../results/plots/scatter_smooth_freq_{result_name}.png', dpi=400)
    # freq-bias stopwords+smooth
    fig_, ax_ = scatter_plt(
        dat_sw, x_var='freq', y_var=m, smooth=True, frac=0.4)
    fig_.savefig(f'../results/plots/scatter_sw_freq_{result_name}.png', dpi=400)
    # freq-bias all words boxplot
    fig_, ax_ = boxplots_plt(
                        dat, x_var='freq', y_var=m, bins=limits)
    fig_.savefig(f'../results/plots/boxplots_freq_{result_name}.png', dpi=400)



#%% Pairs plot

def pairs_plt(data, n_sample=None, seed=123):
    """
    Density plots in diagonales, scatter plots in lower diagonal
    """
    if n_sample:
        data = data.sample(n_sample, random_state=seed)
    features = ['dppmi', 'order2', 'cosine_glove', 'cosine_w2v']
    n_features = len(features)
    fig, axs = plt.subplots(n_features, n_features, figsize=(10,10)
                            ,sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if i == j:
                axs[i][j].hist(data[f2], 50, density=True, alpha=0.5)
                axs[i][j].axvline(0, ls='--', color='black', linewidth=0.5)
                axs[i][j].set_xlabel(f2)
            if i > j:
                axs[i][j].scatter(data[f2], data[f1], s=0.5)
                axs[i][j].axhline(0, ls='--', color='black', linewidth=0.5)
                axs[i][j].axvline(0, ls='--', color='black', linewidth=0.5)
                axs[i][j].set_xlabel(f2)
                axs[i][j].set_ylabel(f1)
    return fig, axs

result_name = f"C{CORPUS_ID}_{TARGET_A}-{TARGET_B}"
fig_, axs_ = pairs_plt(dat, n_sample=20_000)
fig_.savefig(f'../results/plots/pairs_{result_name}.png', dpi=400)
