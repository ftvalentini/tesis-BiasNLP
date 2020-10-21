import numpy as np
import pandas as pd
import os
import sys
import datetime
import re
import argparse

from utils.corpora import load_vocab
from metrics.glove import bias_byword


def main(vocab_file, matrix_file, target_a, target_b):

    print("Loading input data...")
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    embed_matrix = np.load(matrix_file)

    print("Getting words lists...")
    words_lists = dict()
    for f in os.listdir('words_lists'):
        nombre = os.path.splitext(f)[0]
        words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]
    words_a = words_lists[target_a]
    words_b = words_lists[target_b]
    words_context = [w for w in str2count.keys() if w not in words_a + words_b]


    print("Computing bias wrt each context word...")
    result = bias_byword(
        embed_matrix, words_a, words_b, words_context, str2idx, str2count)

    print("Saving results in csv...")
    results_name = re.search("^.+/(.+C\d+)-.+$", matrix_file).group(1)
    outfile = f'results/csv/biasbyword_{results_name}_{target_a}-{target_b}.csv'
    result.to_csv(outfile, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--vocab', type=str, required=True)
    required.add_argument('--matrix', type=str, required=True)
    required.add_argument('--a', type=str, required=True)
    required.add_argument('--b', type=str, required=True)

    args = parser.parse_args()

    print("START -- ", datetime.datetime.now())
    main(args.vocab, args.matrix, args.a, args.b)
    print("END -- ", datetime.datetime.now())
