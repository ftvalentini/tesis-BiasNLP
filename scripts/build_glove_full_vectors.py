import numpy as np
import os, struct, pickle, datetime, sys
from tqdm import tqdm

from utils.corpora import load_vocab


def read_glove_vectors(vocab_file, embed_file):
    """
    Read weights of glove .bin embed_file
    Returns (W, b_w, U, b_u) [word vectors, w biases, context vectors, biases]
    Notas:
    hay dos vectores por word + 2 vectores constantes (biases)
    el bin esta ordenado por indice -- tienen vector dim d+1 (bias)
    see https://github.com/mebrunet/understanding-bias/blob/master/src/GloVe.jl#L39
    (los indices del vocab empiezan en 1 --> columna 0 queda con nan)
    """
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    V = len(str2idx)  # vocab size
    n_weights = os.path.getsize(embed_file) // 8 # 8 bytes per double
    embed_dim = (n_weights - 2*V) // (2*V)
    # init arrays
    W = np.full((embed_dim, V+1), np.nan) # word vectors
    b_w = np.full(V+1, np.nan) # word vectors' biases
    U = np.full((embed_dim, V+1), np.nan) # context vectors
    b_u = np.full(V+1, np.nan) # context vectors' biases
    # first word index (should be 1)
    first_i = min(list(idx2str.keys()))
    with open(embed_file, 'rb') as f:
        # read word vectors and bias
        for i in range(first_i, V+1):
            W[:,i] = struct.unpack('d'*embed_dim, f.read(8*embed_dim))
            b_w[i] = struct.unpack('d'*1, f.read(8*1))[0] # 'd' for double
        # read context vectors and bias
        for i in range(first_i, V+1):
            U[:,i] = struct.unpack('d'*embed_dim, f.read(8*embed_dim))
            b_u[i] = struct.unpack('d'*1, f.read(8*1))[0] # 'd' for double
    return W, b_w, U, b_u


def main(vocabfile, embedfile, outfile):
    print(f'Input vocab file is {vocabfile}')
    print(f'Input embedding file is {embedfile}')
    print(f'Output pkl file is {outfile}')
    vectors = read_glove_vectors(vocabfile, embedfile)
    with open(outfile, 'wb') as f:
        pickle.dump(vectors, f)


if __name__ == "__main__":
    print("START -- ", datetime.datetime.now())
    vocabfile = sys.argv[1]
    embedfile = sys.argv[2]
    outfile = sys.argv[3]
    main(vocabfile, embedfile, outfile)
    print("END -- ", datetime.datetime.now())
