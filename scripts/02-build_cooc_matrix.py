import numpy as np
import sys
import scipy.sparse
from tqdm import tqdm
from ctypes import Structure, c_int, c_double, sizeof

from utils.corpora import load_vocab


class CREC(Structure):
    """c++ class to read triples (idx, idx, cooc) from GloVe binary file
    """
    _fields_ = [('idx1', c_int),
                ('idx2', c_int),
                ('value', c_double)]


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
    # init sparse matrix VxV (los indices del vocab empiezan en 1)
    C = scipy.sparse.dok_matrix((V+1, V+1), dtype=np.float32)
    # pbar = tqdm(total=V)
    k = min(list(idx2str.keys()))  # contador
    k_max = max(list(idx2str.keys()))
    # open bin file and store coocs in C
    with open(cooc_file, 'rb') as f:
        # read and overwrite into cr while there is data
        cr = CREC()
        while (f.readinto(cr) == size_crec):
            C[cr.idx1, cr.idx2] = cr.value
            if cr.idx1 > k:
                k += 1
                print(f'{k} of {k_max} DONE')
                # pbar.update(1)
    # pbar.close()
    return C.tocsr()


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
    path_dir = path.abspath(path.join(__file__ , path.pardir, path.pardir))
    args = dict()
    args["vocabfile"] = path.join(path_dir, options.vocabfile)
    args["coocfile"] = path.join(path_dir, options.coocfile)
    args["outfile"] = path.join(path_dir, options.outfile)
    print(f'Input vocab file is {args["vocabfile"]}')
    print(f'Input cooc file is {args["coocfile"]}')
    print(f'Output matrix file is {args["outfile"]}')
    C = build_cooc_matrix(args["vocabfile"], args["coocfile"])
    scipy.sparse.save_npz(args["outfile"], C)


if __name__ == "__main__":
    main(sys.argv[1:])
