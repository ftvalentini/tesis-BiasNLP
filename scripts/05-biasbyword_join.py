import numpy as np
import pandas as pd
import datetime
import sys
import re
import argparse
import nltk
import scipy.stats as stats

from functools import reduce
from pathlib import Path
from nltk.corpus import stopwords


def main(corpus_id, target_a, target_b):

    # input files
    raiz = Path("results/csv")
    file_dppmi =  raiz / f"biasbyword_dppmi-C{corpus_id}_{target_a}-{target_b}.csv"
    file_order2 = raiz / f"biasbyword_order2-C{corpus_id}_{target_a}-{target_b}.csv"
    file_glove = raiz / f"biasbyword_glove-C{corpus_id}_{target_a}-{target_b}.csv"
    file_w2v = raiz / f"biasbyword_w2v-C{corpus_id}_{target_a}-{target_b}.csv"

    # read data
    df_dppmi = pd.read_csv(file_dppmi)
    df_order2 = pd.read_csv(file_order2)
    df_glove = pd.read_csv(file_glove)
    df_w2v = pd.read_csv(file_w2v)

    # rename/drop columns
    we_cols = \
        ['cosine_a','cosine_b','dot_a','dot_b','diff_cosine','diff_dot','norm']
    df_glove.rename(columns={col: col+"_glove" for col in we_cols}, inplace=True)
    df_w2v.rename(columns={col: col+"_w2v" for col in we_cols}, inplace=True)
    df_order2.rename(columns={col: col+"_ppmi" for col in we_cols}, inplace=True)
    df_order2.rename(columns={"nnz": "nnz_ppmi"}, inplace=True)
    df_w2v.drop(columns=['freq'], inplace=True)

    # merge
    df_list = [df_dppmi, df_order2, df_glove, df_w2v]
    df = reduce(
        lambda a,b: pd.merge(a, b, on=['word','idx'], how='inner'), df_list)

    # cols to keep
    ppmi_cols = [c+"_ppmi" for c in we_cols]
    w2v_cols = [c+"_w2v" for c in we_cols]
    glove_cols = [c+"_glove" for c in we_cols]
    get_cols = [
        'idx','word','freq'
        ,'dppmi','pvalue','count_total','count_context_a','count_context_b'
        ,'pmi_a','pmi_b','log_oddsratio','lower','upper'
        ,'nnz_ppmi']
    get_cols += ppmi_cols
    get_cols += w2v_cols
    get_cols += glove_cols
    df = df[get_cols]

    # oddsratio
    df['oddsratio'] = np.exp(df['log_oddsratio'])

    # rankings
    # # abs(log_oddsratio) solo si p-value < 0.01
    # lor_abs = df.loc[df['pvalue'] < 0.01]['log_oddsratio'].abs()
    # rank_lor_abs = stats.rankdata(lor_abs, "average")/len(lor_abs)
    # df.loc[df['pvalue'] < 0.01, 'rank_lor_abs'] = rank_lor_abs
    # df.loc[df['pvalue'] >= 0.01, 'rank_lor_abs'] = 0.0
    def add_ranking_col(df, colname):
        """Add column with 'rank_' prefix with the rank of the absolute value"""
        col_abs = df[colname].abs()
        newname = 'rank_'+colname+'_abs'
        df[newname] = stats.rankdata(col_abs, "average")/len(col_abs)
        return df

    df = add_ranking_col(df, "dppmi")
    df = add_ranking_col(df, "diff_cosine_ppmi")
    df = add_ranking_col(df, "diff_cosine_w2v")
    df = add_ranking_col(df, "diff_cosine_glove")

    # SW indicator
    words_rm = [
        'he',"he's", 'him','his','himself','she',"she's",'her','hers','herself']
    sw = set(stopwords.words('english')) - set(words_rm)
    df['sw'] = np.where(df['word'].isin(sw), 1, 0)

    # save data
    outfile = raiz / f'biasbyword_full-C{corpus_id}_{target_a}-{target_b}.csv'
    df.to_csv(outfile, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--corpus', type=int, required=True)
    required.add_argument('--a', type=str, required=True)
    required.add_argument('--b', type=str, required=True)

    args = parser.parse_args()

    print("START -- ", datetime.datetime.now())
    main(args.corpus, args.a, args.b)
    print("END -- ", datetime.datetime.now())
