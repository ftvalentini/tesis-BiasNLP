import os, struct
from tqdm import tqdm
from utils.corpora import load_vocab

def get_embeddings(word_list, str2idx
           ,embed_file="embeddings/vectors-C0-V20-W8-D1-D50-R0.05-E100-S1.bin"):
    """ Return dict {word: tuple_embedding} for word in word_list
    """
    vocab_size = len(str2idx)
    n_weights = os.path.getsize(embed_file) // 8 # 8 bytes per double
    embed_dim = (n_weights - 2*vocab_size) // (2*vocab_size)
        # dos vectores por word + 2 vectores constantes (biases)
        # see https://github.com/mebrunet/understanding-bias/blob/master/src/GloVe.jl
    # el bin esta ordenado por indice -- tienen vector dim 50 + 1 bias
    indices = sorted([str2idx[word] for word in word_list])
    idx2str = {str2idx[w]: w for w in word_list}
    embeddings = dict()
    # read 50 weights (double) + 1 bias (double) by word
    with open(embed_file, 'rb') as f:
        # idx start in 1 in idx2str
        for i in tqdm(range(1, vocab_size+1)):
            embedding = struct.unpack('d'*embed_dim, f.read(8*embed_dim))
            bias = struct.unpack('d'*1, f.read(8*1)) # 'd' for double
            if i > indices[-1]:
                break
            if i in indices:
                embeddings[idx2str[i]] = embedding
    return embeddings
