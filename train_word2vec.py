import numpy as np
import pandas as pd

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



# vocab_file = "embeddings/vocab-C3-V20.txt"
# corpus_file = "corpora/enwikiselect.txt"
VOCAB_FILE = "embeddings/vocab-C1-V1.txt"
CORPUS_FILE = "corpora/test.txt"

CORPUS_ID = 0
SG = 0 # 0:cbow, 1:sgns
SIZE = 100
WINDOW = 8
MIN_COUNT = 1 #20
SEED = 1


# create generator of docs
docs = Corpus(CORPUS_FILE)
# train word2vec
model = Word2Vec(
    sentences=docs, size=SIZE, window=WINDOW, min_count=MIN_COUNT, seed=SEED,
    sg=SG)

# test vocab coincide con vocab de glove
str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)
str2count_w2v = {w: model.wv.vocab[w].count for w in model.wv.vocab}
assert str2count_w2v == str2count

# crear matriz con igual formato y orden de palabras que en build_embeddings_matrix
# vectors = model.wv
# ...


# save model just in case?
# model_file = f"embeddings/w2v_C{CORPUS_ID}_V{MIN_COUNT}_W{WINDOW}_D{SIZE}.model"
# model.save(model_file)
