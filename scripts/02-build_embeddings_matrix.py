import numpy as np
import os, struct, sys, datetime
from tqdm import tqdm

from utils.corpora import load_vocab


def build_embeddings_matrix(vocab_file, embed_file):
    """ Build full embedding numpy matrix from data in binary glove file and \
        glove vocab text file
        Size d x V
        (V: vocab size; d: embedding dim)
    """
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    V = len(str2idx)  # vocab size
    n_weights = os.path.getsize(embed_file) // 8 # 8 bytes per double
    embed_dim = (n_weights - 2*V) // (2*V)
        # dos vectores por word + 2 vectores constantes (biases)
        # see https://github.com/mebrunet/understanding-bias/blob/master/src/GloVe.jl
    # el bin esta ordenado por indice -- tienen vector dim d+1 (bias)
    # init array (los indices del vocab empiezan en 1 --> columna 0 queda con nan)
    M = np.full((embed_dim, V+1), np.nan)
    first_i = min(list(idx2str.keys())) # first word index (should be 1)
    # read weights (double) + 1 bias (double) by word
    with open(embed_file, 'rb') as f:
        for i in tqdm(range(first_i, V+1)):
            embedding = struct.unpack('d'*embed_dim, f.read(8*embed_dim))
            bias = struct.unpack('d'*1, f.read(8*1)) # 'd' for double
            M[:,i] = embedding
    return M


def main(argv):
    from optparse import OptionParser
    from os import path
    usageStr = """
    python build_embeddings_matrix.py -v <vocabfile.txt> -e <embed.bin> -o <outfile.npz>'
    Files relative to root project directory
    """
    parser = OptionParser(usageStr)
    parser.add_option('-v', '--vocabfile', dest='vocabfile', type='string')
    parser.add_option('-e', '--embedfile', dest='embedfile', type='string')
    parser.add_option('-o', '--outfile', dest='outfile', type='string')
    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    path_dir = path.abspath(path.join(__file__ , path.pardir, path.pardir))
    args = dict()
    args["vocabfile"] = path.join(path_dir, options.vocabfile)
    args["embedfile"] = path.join(path_dir, options.embedfile)
    args["outfile"] = path.join(path_dir, options.outfile)
    print(f'Input vocab file is {args["vocabfile"]}')
    print(f'Input embedding file is {args["embedfile"]}')
    print(f'Output matrix file is {args["outfile"]}')
    M = build_embeddings_matrix(args["vocabfile"], args["embedfile"])
    np.save(args["outfile"], M)


if __name__ == "__main__":
    print("START -- ", datetime.datetime.now())
    main(sys.argv[1:])
    print("END -- ", datetime.datetime.now())
