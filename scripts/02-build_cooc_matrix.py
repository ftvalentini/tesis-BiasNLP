import numpy as np
import scipy.sparse
import sys
import os
import datetime
import array
from tqdm import tqdm
from ctypes import Structure, c_int, c_double, sizeof

from utils.corpora import load_vocab


class CREC(Structure):
    """c++ class to read triples (idx, idx, cooc) from GloVe binary file
    """
    _fields_ = [('idx1', c_int),
                ('idx2', c_int),
                ('value', c_double)]


class IncrementalCOOMatrix:
    """class to create scipy.sparse.coo_matrix
    """

    def __init__(self, shape, dtype=np.double):
        self.dtype = dtype
        self.shape = shape
        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array('d')

    def append(self, i, j, v):
        m, n = self.shape
        if (i >= m or j >= n):
            raise Exception('Index out of bounds')
        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):
        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)
        return scipy.sparse.coo_matrix((data, (rows, cols)),
                                       shape=self.shape)


def build_cooc_matrix(vocab_file, cooc_file):
    """
    Build full coocurrence matrix from cooc. data in binary glove file and \
    glove vocab text file
    Row and column indices are numeric indices from vocab_file
    There must be (i,j) for every (j,i) such that C[i,j]=C[j,i]
    """
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    V = len(str2idx)  # vocab size
    size_crec = sizeof(CREC)  # crec: structura de coocucrrencia en Glove
    C = IncrementalCOOMatrix((V+1, V+1))
    # C = scipy.sparse.coo_matrix((V+1, V+1), dtype=np.double)
    # k = min(list(idx2str.keys()))  # contador
    K = os.path.getsize(cooc_file) / size_crec # total de coocurrencias
    pbar = tqdm(total=K)
    # open bin file and store coocs in C
    with open(cooc_file, 'rb') as f:
        # read and overwrite into cr while there is data
        cr = CREC()
        while (f.readinto(cr) == size_crec):
            C.append(cr.idx1, cr.idx2, cr.value)
            pbar.update(1)
            # C[cr.idx1, cr.idx2] = cr.value
            # if cr.idx1 > k:
                # k += 1
    pbar.close()
    return C.tocoo().tocsr()


def main(argv):
    from optparse import OptionParser
    from os import path
    usageStr = """
    python build_cooc_matrix.py -v <vocabfile.txt> -c <coocfile.bin> -o <outfile.npz>'
    Files relative to root project directory
    """
    parser = OptionParser(usageStr)
    parser.add_option('-v', '--vocabfile', dest='vocabfile', type='string')
    parser.add_option('-c', '--coocfile', dest='coocfile', type='string')
    parser.add_option('-o', '--outfile', dest='outfile', type='string')
    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    path_dir = path.abspath(path.join(__file__, path.pardir, path.pardir))
    args = dict()
    args["vocabfile"] = path.join(path_dir, options.vocabfile)
    args["coocfile"] = path.join(path_dir, options.coocfile)
    args["outfile"] = path.join(path_dir, options.outfile)
    print(f'Input vocab file is {args["vocabfile"]}')
    print(f'Input cooc file is {args["coocfile"]}')
    print(f'Output matrix file is {args["outfile"]}')
    print("BUILDING MATRIX -- ", datetime.datetime.now())
    C = build_cooc_matrix(args["vocabfile"], args["coocfile"])
    print("SAVING MATRIX -- ", datetime.datetime.now())
    scipy.sparse.save_npz(args["outfile"], C)


if __name__ == "__main__":
    print("START -- ", datetime.datetime.now())
    main(sys.argv[1:])
    print("DONE -- ", datetime.datetime.now())


# ### FAILED:
#     # read whole data at once
#     with open(cooc_file, 'rb') as f:
#         rows, cols, cells = list(zip(*np.frombuffer(f.read(), CREC)))
#     C = scipy.sparse.coo_matrix((cells, (rows, cols)))

# # TEST:
# vocab_file = "../embeddings/vocab-C1-V1.txt"
# cooc_file = "../embeddings/cooc-C1-V1-W8-D0.bin"
# out_file = "../embeddings/cooc-C1-V1-W8-D0.npz"
# M = build_cooc_matrix(vocab_file, cooc_file)
# scipy.sparse.save_npz(out_file, M)
