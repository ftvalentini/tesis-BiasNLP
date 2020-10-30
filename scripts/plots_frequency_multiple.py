import numpy as np
import pandas as pd
import datetime
import sys
import re

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils.corpora import load_vocab


def create_df(file_glove, file_w2v):
    """
    Create DataFrame with tidy results
    """
    # read data
    df_glove = pd.read_csv(file_glove)
    df_w2v = pd.read_csv(file_w2v)
    # rename/drop columns
    df_glove.rename(columns={"rel_cosine_similarity": "cosine_glove"}, inplace=True)
    df_w2v.rename(columns={"rel_cosine_similarity": "cosine_w2v"}, inplace=True)
    df_w2v.drop(columns=['freq'], inplace=True)
    # merge
    df = pd.merge(df_glove, df_w2v, on=['word','idx'], how='inner')
    # SW indicator
    words_rm = [
        'he',"he's", 'him','his','himself','she',"she's",'her','hers','herself']
    sw = set(stopwords.words('english')) - set(words_rm)
    df['sw'] = np.where(df['word'].isin(sw), 1, 0)
    # cols to keep
    get_cols = ['idx','word','freq','cosine_glove','cosine_w2v','sw']
    return df[get_cols]


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
    plt.xlabel(f"log10({x_var})")
    labels_ypos = ax.get_ylim()[0] + .001
    for i in range(len(nobs)):
        ax.text(
            i, labels_ypos, nobs[i], horizontalalignment='center', size='small'
            , color='black', weight='semibold')
    return fig, ax


def make_plots(corpus_n, target_a="HE", target_b="SHE"):
    """
    Make 4 plots for cosine bias glove and w2v
    """
    # input files
    path_vocab = Path("../embeddings")
    path_csv = Path("../results/csv")
    path_plots = Path("../results/plots")
    file_glove = path_csv / f"biasbyword_glove-C{corpus_n}_{target_a}-{target_b}.csv"
    file_w2v = path_csv / f"biasbyword_w2v-C{corpus_n}_{target_a}-{target_b}.csv"
    vocab_file = path_vocab / f"vocab-C{corpus_n}-V20.txt"
    # read data and vocab
    dat = create_df(file_glove, file_w2v)
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    # dat only with SW
    dat_sw = dat.loc[dat['sw'] == 1]
    # counts he-she ratio
    ratio = str2count[target_a.lower()] / str2count[target_b.lower()]
    metrics = ['cosine_glove', 'cosine_w2v']
    for m in metrics:
        result_name = f"{m}-C{corpus_n}_{target_a}-{target_b}"
        title_name = f"Freq. {target_a}/{target_b} = {ratio:.2f}"
        # freq-bias all words
        fig_, ax_ = scatter_plt(dat, x_var='freq', y_var=m)
        plt.title(title_name)
        fig_.savefig(path_plots / f'scatter_freq_{result_name}.png', dpi=400)
        # freq-bias sample+smooth
        fig_, ax_ = scatter_plt(dat, x_var='freq', y_var=m, n_sample=20_000, smooth=True)
        plt.title(title_name)
        fig_.savefig(path_plots / f'scatter_smooth_freq_{result_name}.png', dpi=400)
        # freq-bias stopwords+smooth
        fig_, ax_ = scatter_plt(dat_sw, x_var='freq', y_var=m, smooth=True, frac=0.4)
        plt.title(title_name)
        fig_.savefig(path_plots / f'scatter_sw_freq_{result_name}.png', dpi=400)
        # freq-bias all words boxplot
        fig_, ax_ = boxplots_plt(dat, x_var='freq', y_var=m, bins=[1,2,3,4,5,6,8])
        plt.title(title_name)
        fig_.savefig(path_plots / f'boxplots_freq_{result_name}.png', dpi=400)


def save_grid(plt_name, metric_name, corpus_ids):
    """
    Make and save grids of already saved pngs
    """
    # file names
    path_plots = Path("../results/plots")
    files = [path_plots / f"{plt_name}_freq_{metric_name}-C{i}_HE-SHE.png" \
                                                            for i in corpus_ids]
    # read images
    images = [plt.imread(str(f)) for f in files]
    # make grid
    fig_, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    for img, ax in zip(images, axes.flat):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # save
    fig_.savefig(path_plots / f"grid_{plt_name}_{metric_name}_HE-SHE.png", dpi=600)



# RUN
for i in range(4,9):
    make_plots(corpus_n=i, target_a="HE", target_b="SHE")


# Save grids
ids = [4, 5, 6, 7, 8, 3]
plots = ["boxplots", "scatter", "scatter_smooth", "scatter_sw"]
metrics = ["cosine_glove", "cosine_w2v"]
for p in plots:
    for m in metrics:
        save_grid(p, m, ids)
