import numpy as np
import pandas as pd
import os
import datetime
import re
import argparse

import scipy.sparse


def build_ppmi_matrix(matrix_file):

    print("Loading cooc. matrix...")
    M = scipy.sparse.load_npz(matrix_file)

    print("Computing PPMI...")
    # ver https://www.aclweb.org/anthology/Q15-1016.pdf seccion 2.1
    nrows, ncols = M.shape
    D = M.count_nonzero()
    colSums = M.sum(axis=0).ravel()
    rowSums = M.sum(axis=1).ravel()
    colDivs = 1.0 / np.array(colSums)[0]
    rowDivs = 1.0 / np.array(rowSums)[0]
    M_ppmi = M * D
    M_ppmi = M_ppmi.dot(scipy.sparse.diags(colDivs))
    M_ppmi = scipy.sparse.diags(rowDivs).dot(M_ppmi)
    M_ppmi.data = np.log(M_ppmi.data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--matrix', type=str, required=True)
    required.add_argument('--outfile', type=str, required=True)

    args = parser.parse_args()

    print("START -- ", datetime.datetime.now())
    mat = build_ppmi_matrix(args.matrix)
    print("Saving PPMI matrix...")
    scipy.sparse.save_npz(args.outfile, mat)
    print("END -- ", datetime.datetime.now())
