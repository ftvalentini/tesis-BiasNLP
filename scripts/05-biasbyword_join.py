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
    df_glove.rename(columns={"rel_cosine_similarity": "cosine_glove"}, inplace=True)
    df_w2v.rename(columns={"rel_cosine_similarity": "cosine_w2v"}, inplace=True)
    df_w2v.drop(columns=['freq'], inplace=True)

    # merge
    df_list = [df_dppmi, df_order2, df_glove, df_w2v]
    df = reduce(
        lambda a,b: pd.merge(a, b, on=['word','idx'], how='inner'), df_list)

    # cols to keep
    get_cols = [
        'idx','word','freq'
        ,'dppmi','order2','cosine_glove','cosine_w2v'
        ,'pvalue','count_total','count_context_a','count_context_b'
        ,'pmi_a','pmi_b','log_oddsratio','lower','upper'
        ,'cosines_a', 'cosines_b', 'dots_a', 'dots_b', 'order2_dot']
    df = df[get_cols]

    # oddsratio
    df['oddsratio'] = np.exp(df['log_oddsratio'])

    # rankings
    # # abs(log_oddsratio) solo si p-value < 0.01
    # lor_abs = df.loc[df['pvalue'] < 0.01]['log_oddsratio'].abs()
    # rank_lor_abs = stats.rankdata(lor_abs, "average")/len(lor_abs)
    # df.loc[df['pvalue'] < 0.01, 'rank_lor_abs'] = rank_lor_abs
    # df.loc[df['pvalue'] >= 0.01, 'rank_lor_abs'] = 0.0
    # abs(dppmi)
    dppmi_abs = df['dppmi'].abs()
    df['rank_dppmi_abs'] = stats.rankdata(dppmi_abs, "average")/len(dppmi_abs)
    # order2
    order2_abs = df['order2'].abs()
    df['rank_order2_abs'] = stats.rankdata(order2_abs, "average")/len(order2_abs)
    # glove
    glove_abs = df['cosine_glove'].abs()
    df['rank_glove_abs'] = stats.rankdata(glove_abs, "average")/len(glove_abs)
    # w2v
    w2v_abs = df['cosine_w2v'].abs()
    df['rank_w2v_abs'] = stats.rankdata(w2v_abs, "average")/len(w2v_abs)

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
