import numpy as np
import pandas as pd
import os
import datetime
import re
import argparse

import scipy.sparse

from utils.corpora import load_vocab
from metrics.cooc import dpmi_byword, order2_byword

WORD_MIN_COUNT = 20


def main(vocab_file, matrix_file, target_a, target_b):

    print("Loading input data...")
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    cooc_matrix = scipy.sparse.load_npz(matrix_file)

    print("Getting words lists...")
    words_lists = dict()
    for f in os.listdir('words_lists'):
        nombre = os.path.splitext(f)[0]
        words_lists[nombre] = [line.strip() for line in open('words_lists/' + f, 'r')]
    words_a = words_lists[target_a]
    words_b = words_lists[target_b]
    words_context = [w for w, freq in str2count.items() if \
                        w not in words_a + words_b and freq >= WORD_MIN_COUNT]

    print("Computing DPMI bias wrt each context word...")
    res_dpmi = dpmi_byword(
                        cooc_matrix, words_a, words_b, words_context, str2idx)

    print("Saving DPMI results in csv...")
    results_name = re.search("^.+/cooc-(C\d+)-.+$", matrix_file).group(1)
    outfile = \
        f'results/csv/biasbyword_dpmi-{results_name}_{target_a}-{target_b}.csv'
    res_dpmi.to_csv(outfile, index=False)

    print("Computing order2 bias wrt each context word...")
    res_order2 = order2_byword(
            cooc_matrix, words_a, words_b, words_context, str2idx)

    print("Saving order2 results in csv...")
    outfile = \
        f'results/csv/biasbyword_order2-{results_name}_{target_a}-{target_b}.csv'
    res_order2.to_csv(outfile, index=False)


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
