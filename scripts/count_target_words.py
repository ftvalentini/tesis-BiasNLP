import sys
import pickle
from pathlib import Path
from tqdm import tqdm


def counts_dict(corpus, word_a, word_b):
    """
    Read each doc from corpus.txt file and count appearances of word_a, word_b
    Returns:
        dict {i_doc: (count_a, count_b)}
    """
    corpus_name = Path(corpus).stem
    # get total number of docs
    metadata_file = f"corpora/{corpus_name}.meta.txt"
    meta = {}
    with open(metadata_file) as f:
        for line in f:
            (key, val) = line.split(": ")
            meta[key] = val
    total_docs = int(meta['num_documents'])
    # get counts for each doc
    i = 0
    counts = dict()
    pbar = tqdm(total=total_docs)
    with open(corpus, "r", encoding="utf-8") as f:
        while True:
            tokens = f.readline().strip().split()
            a_count, b_count = tokens.count(word_a), tokens.count(word_b)
            counts[i] = (a_count, b_count)
            i += 1
            pbar.update(1)
            if not tokens:
                break
        pbar.close()
    return counts


if __name__ == "__main__":
    # param
    corpus = sys.argv[1]
    word_a = sys.argv[2]
    word_b = sys.argv[3]
    # get counts
    counts = counts_dict(corpus, word_a, word_b)
    # save as pickle
    corpus_name = Path(corpus).stem
    with open(f"corpora/{corpus_name}_counts_{word_a}-{word_b}.pkl", 'wb') as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
