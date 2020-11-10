import numpy as np
import pandas as pd
import datetime
import sys
import re

import seaborn as sns
import matplotlib.pyplot as plt
import nltk

from pathlib import Path
# from nltk.corpus import stopwords
# from statsmodels.nonparametric.smoothers_lowess import lowess

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
    # newwords indicator
    new_words = ['c'+str(i) for i in range(1,7)]
    df['new_word'] = np.where(df['word'].isin(new_words), 1, 0)
    # cols to keep
    get_cols = ['idx','word','freq','cosine_glove','cosine_w2v','new_word']
    return df[get_cols]


# file_glove = "../results/csv/biasbyword_glove-C9_HE-SHE.csv"
# file_w2v = "../results/csv/biasbyword_w2v-C9_HE-SHE.csv"
# dat = create_df(file_glove, file_w2v)
# dat.query("word in ('he','she','t1','t2','c1','c2','c3','c4','c5','c6')")

str2idx, idx2str, str2count = load_vocab("../embeddings/vocab-C9-V20.txt")

tmp = {k:v for k,v in str2count.items() if k in
                            ('he','she','t1','t2','c1','c2','c3','c4','c5','c6')}
pd.DataFrame({'word': list(tmp.keys()), 'freq': list(tmp.values())})


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
    data_sw = data[data['new_word'] == 1]
    data_resto = data[data['new_word'] == 0]
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xlim(left=10, right=10**8)
    plt.scatter(x_var, y_var, linewidth=0, c='darkolivegreen', s=4, data=data_resto)
    plt.scatter(x_var, y_var, linewidth=0, c='firebrick', s=30, data=data_sw)
    if smooth:
        smooth_data = lowess(data[y_var], np.log10(data[x_var]), frac=frac)
        line = ax.plot(
            10**smooth_data[:,0], smooth_data[:,1], color='black', lw=1.0, ls='--')
    ax.axhline(0, ls='--', color='gray', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
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
    # counts he-she ratio
    ratio = str2count[target_a.lower()] / str2count[target_b.lower()]
    metrics = ['cosine_glove', 'cosine_w2v']
    for m in metrics:
        result_name = f"{m}-C{corpus_n}_{target_a}-{target_b}"
        title_name = f"Freq. {target_a}/{target_b} = {ratio:.2f}"
        # freq-bias all words
        fig, ax = scatter_plt(dat, x_var='freq', y_var=m)
        plt.title(title_name)
        fig.savefig(path_plots / f'scatter_freq_{result_name}.png', dpi=400)


make_plots(corpus_n=9, target_a="HE", target_b="SHE")
make_plots(corpus_n=9, target_a="T1", target_b="T2")



def make_pairs_plt(target_a="HE", target_b="SHE", n_sample=None, seed=123):
    """
    Density plots in diagonales, scatter plots in lower diagonal
    """
    path_csv = Path("../results/csv")
    path_plots = Path("../results/plots")
    # Corpus con newwords
    file_glove = path_csv / f"biasbyword_glove-C9_{target_a}-{target_b}.csv"
    file_w2v = path_csv / f"biasbyword_w2v-C9_{target_a}-{target_b}.csv"
    df_new = create_df(file_glove, file_w2v)
    # Corpus original
    file_orig = path_csv / f"biasbyword_full-C3_{target_a}-{target_b}.csv"
    df_orig = pd.read_csv(file_orig)
    # Merge
    df_new.rename(columns={"cosine_glove": "cosine_glove_new"}, inplace=True)
    df_new.rename(columns={"cosine_w2v": "cosine_w2v_new"}, inplace=True)
    df_new.drop(columns=['freq', 'idx'], inplace=True)
    data = pd.merge(df_new, df_orig, "inner", on=["word"])
    # sample
    if n_sample:
        data = data.sample(n_sample, random_state=seed)
    features = ["cosine_glove", "cosine_glove_new", "cosine_w2v", "cosine_w2v_new"]
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


fig_, axs_ = make_pairs_plt(target_a="HE", target_b="SHE", n_sample=20_000)
