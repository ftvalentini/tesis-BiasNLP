import numpy as np
import pandas as pd
import argparse

from pathlib import Path
from gensim.models.word2vec import Word2Vec

from scripts.utils.corpora import load_vocab


class Corpus:
    """
    Helper iterator that yields documents (doc: list of str)
    Needed so that gensim can read docs from disk on the fly
    """
    def __init__(self, corpus_file):
        """
        corpus_file: a txt with one document per line and tokens separated
        by whitespace
        """
        self.corpus_file = corpus_file
    def __iter__(self):
        for line in open(self.corpus_file, encoding="utf-8"):
            # returns each doc as a list of tokens
            yield line.split()


def train_w2v(corpus_file, size, window, min_count, seed, sg):
    """
    Returns w2v gensim trained model
    Params:
        - min_count: min word frequency
        - sg: 1 if skipgram -- 0 if cbow
    """
    # create generator of docs
    docs = Corpus(CORPUS_FILE)
    # train word2vec
    model = Word2Vec(
        sentences=docs, size=size, window=window, min_count=min_count
        , seed=seed, sg=sg)
    return model


def w2v_to_array(w2v_model, str2idx):
    """
    Convert w2v vectors to np.array with dim DIMx(V+1), using column indices
    of str2idx of vocab_file produced by GloVe
    """
    vectors = w2v_model.wv
    D = model.wv.vector_size
    V = len(str2idx)
    M = np.full((D, V+1), np.nan)
    for w, i in str2idx.items():
        M[:,i] = vectors[w]
    return M


def main(corpus_id, corpus_file, vocab_file, outdir, **kwargs_w2v):
    """
    Train w2v, save w2v model and save embeddings matrix
    """
    # load vocab
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    # train vectors
    model = train_w2v(corpus_file, size, window, min_count, seed, sg):
    # TEST vocab coincide con vocab de glove
    str2count_w2v = {w: model.wv.vocab[w].count for w in model.wv.vocab}
    assert str2count_w2v == str2count
    # w2v vectors as np.array
    embed_matrix = w2v_to_array(model, str2idx)
    # save w2v model
    kw = kwargs_w2v
    basename =
        f"w2v_C{corpus_id}_V{kw['min_count']}_W{kw['window']}_D{kw['size']}_SG{kw['sg']}"
    model_file = Path(outdir) / "embeddings" / f"{basename}.model"
    model.save(model_file)
    # save embeddings array
    matrix_file = Path(outdir) / "embeddings" / f"{basename}.npz"
    np.save(matrix_file, embed_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--id', type=int, required=True)
    required.add_argument('--corpus', type=str, required=True)
    required.add_argument('--vocab', type=str, required=True)
    required.add_argument('--outdir', type=str, required=True)
    required.add_argument('--size', type=int, required=True)
    required.add_argument('--window', type=int, required=True)
    required.add_argument('--count', type=int, required=True)
    required.add_argument('--sg', type=int, required=True)
    required.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    kwargs_w2v = {
        'size': args.size, 'window': args.window, 'min_count': args.count
        , 'sg': args.sg, 'seed': args.seed
        }
    main(args.id, args.corpus, args.vocab, args.outdir, **kwargs_w2v)
