import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from contextlib import ExitStack
from tqdm import tqdm

from utils.corpora import load_vocab


WORD_A = "he"
WORD_B = "she"
RATIOS = [4.0, 2.0, 1.0, 0.5, 0.25]


def main(corpus_file, vocab_file, counts_file, outdir, seed):

    # load vocab
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    # load counts
    with open(counts_file, "rb") as f:
        counts = pickle.load(f)

    # counts as DataFrame
    df = pd.DataFrame({'doc': list(counts.keys()), 'counts': list(counts.values())})
    df['count_a'] = [v[0] for v in df['counts']]
    df['count_b'] = [v[1] for v in df['counts']]
    df['ratio'] = np.divide(df['count_a'], df['count_b'])

    # sample with repl. docs with count_b > count_a (ratio<1)
    df_to_sample = df.query("ratio < 1")
    sampled_docs = df_to_sample.sample(10_000_000, replace=True, random_state=seed)
    sampled_docs = sampled_docs[['doc', 'count_a', 'count_b']]
    # cumulative counts and cumulative ratio of A/B
    sampled_docs['count_a_cum'] = np.cumsum(sampled_docs['count_a'])
    sampled_docs['count_b_cum'] = np.cumsum(sampled_docs['count_b'])
    sampled_docs['count_a_tot'] = sampled_docs['count_a_cum'] + str2count[WORD_A]
    sampled_docs['count_b_tot'] = sampled_docs['count_b_cum'] + str2count[WORD_B]
    sampled_docs['ratio_tot'] = sampled_docs['count_a_tot'] / sampled_docs['count_b_tot']

    # keep only rows of docs to write
    sampled_docs = sampled_docs.loc[sampled_docs['ratio_tot'] >= RATIOS[-1]]
    tot_docs = sampled_docs.shape[0]

    # read base corpus
    print("loading full corpus...\n")
    with open(corpus_file, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    # outfile names
    corpus_name = Path(corpus_file).stem
    outfiles = [corpus_name + "_oversampled_" + str(r) + ".txt" for r in RATIOS]
    outfiles = [Path(outdir) / f for f in outfiles]

    # iterate by row: write doc to file according to ratio
    pbar = tqdm(total=tot_docs)
    with ExitStack() as stack:
        files = [stack.enter_context(open(file, "w", encoding="utf-8")) \
                                                        for file in outfiles]
        for index, row in sampled_docs.iterrows():
            pbar.update(1)
            for r, f in zip(RATIOS, files):
                if row['ratio_tot'] >= r:
                    i_doc = int(row['doc'])
                    f.write(lineas[i_doc])
        pbar.close()

    # append whole corpus at the end of each file
    print("appending whole corpus to each file...")
    with ExitStack() as stack:
        files = [stack.enter_context(open(file, "a", encoding="utf-8")) \
                                                        for file in outfiles]
        for f in files:
            f.writelines(lineas)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_file', type=str)
    parser.add_argument('vocab_file', type=str)
    parser.add_argument('counts_file', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('seed', type=int)

    args = parser.parse_args()

    main(args.corpus_file, args.vocab_file, args.counts_file, args.outdir
        , args.seed)
