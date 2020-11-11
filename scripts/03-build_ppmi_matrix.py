import numpy as np
import pandas as pd
import os
import datetime
import re
import argparse

import scipy.sparse


def build_pmi_matrix(M):
    """
    Build PMI matrix using cooc. matrix
    Param:
        - M: scipy.sparse cooc. matrix
    Notes:
        - ver https://www.aclweb.org/anthology/Q15-1016.pdf seccion 2.1
        - words a,b con cooc(a,b)=0 reciben pmi(a,b)=0
    """
    nrows, ncols = M.shape
    D = M.count_nonzero()
    colSums = M.sum(axis=0).ravel()
    rowSums = M.sum(axis=1).ravel()
    colDivs = 1.0 / np.array(colSums)[0]
    rowDivs = 1.0 / np.array(rowSums)[0]
    M_pmi = M * D
    M_pmi = M_pmi.dot(scipy.sparse.diags(colDivs))
    M_pmi = scipy.sparse.diags(rowDivs).dot(M_pmi)
    M_pmi.data = np.log(M_pmi.data)
    return M_pmi


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--matrix', type=str, required=True)
    required.add_argument('--outfile', type=str, required=True)

    args = parser.parse_args()

    print("START -- ", datetime.datetime.now())
    print("Loading cooc. matrix...")
    M = scipy.sparse.load_npz(args.matrix)
    print("Computing PMI matrix...")
    mat = build_pmi_matrix(M)
    print("Saving PMI matrix...")
    scipy.sparse.save_npz(args.outfile, mat)
    print("END -- ", datetime.datetime.now())
