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

# corpus_file="../corpora/enwikiselect.txt"
# vocab_file="../embeddings/vocab-C3-V1.txt"
# counts_file="../corpora/enwikiselect_counts_he-she.pkl"
# outdir="E:\\tesis-BiasNLP\\corpora"
# seed = 88

def main(corpus_file, vocab_file, counts_file, outdir, seed):

    corpus_name = Path(corpus_file).stem
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

    # shuffle docs with count_b < count_a (ratio>1)
    shuffled_docs = df.query("ratio > 1")
    shuffled_docs = shuffled_docs.sample(frac=1, replace=False, random_state=seed)
    shuffled_docs = shuffled_docs[['doc', 'count_a', 'count_b']]
    # cumulative counts and cumulative ratio of A/B without cum. docs
    shuffled_docs['count_a_cum'] = np.cumsum(shuffled_docs['count_a'])
    shuffled_docs['count_b_cum'] = np.cumsum(shuffled_docs['count_b'])
    shuffled_docs['count_a_tot'] = str2count[WORD_A] - shuffled_docs['count_a_cum']
    shuffled_docs['count_b_tot'] = str2count[WORD_B] - shuffled_docs['count_b_cum']
    shuffled_docs['ratio_tot'] = shuffled_docs['count_a_tot'] / shuffled_docs['count_b_tot']

    # keep only rows of docs to remove
    shuffled_docs = shuffled_docs.loc[shuffled_docs['ratio_tot'] >= RATIOS[-1]]
    tot_docs = shuffled_docs.shape[0]

    # read base corpus
    print("Loading full corpus...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    # iterate by ratio
    for ratio in RATIOS:
        print(f"Slicing lines of ratio={ratio}")
        indices_to_rm = sorted(
                    shuffled_docs.loc[shuffled_docs['ratio_tot'] >= ratio]['doc'])
        indices_to_write = set(range(len(lineas))) - set(indices_to_rm)
        lineas_to_write = [lineas[i] for i in indices_to_write]
        outfile = corpus_name + "_undersampled_" + str(ratio) + ".txt"
        outfile = Path(outdir) / outfile
        print(f"Writing lines of ratio={ratio}")
        with open(outfile, "w", encoding="utf-8") as f:
            f.writelines(lineas_to_write)


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
