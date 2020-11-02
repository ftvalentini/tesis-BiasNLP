import numpy as np
import datetime
import argparse

from tqdm import tqdm
from blist import blist

from utils.corpora import load_vocab


TARGET_A = "HE"
TARGET_B = "SHE"

# corpusfile = "../corpora/test2.txt"
# outfile = "prueba.txt"

def main(corpusfile, vocabfile, outfile, seed):

    print("Reading and splitting corpus...")
    n_docs = sum(1 for line in open(corpusfile, 'r', encoding="utf-8"))
    words = blist()
    with open(corpusfile, encoding="utf-8") as f:
        for doc in tqdm(f, total=n_docs):
            words_doc = doc.split(" ")
            words.extend(words_doc)

    print("Creating list of new words...")
    str2idx, idx2str, str2count = load_vocab(vocabfile)
    # new words with frecuencies (t1, t2, c1, c2, ..., c6)
    word2freq = {'c'+str(i): 2*(10**i) for i in range(1,7)}
    word2freq['t1'] = str2count[TARGET_A.lower()]
    word2freq['t2'] = str2count[TARGET_B.lower()]
    word2freq_low = {k: v for k, v in word2freq.items() if v < 500_000}
    word2freq_high = {k: v for k, v in word2freq.items() if k not in word2freq_low}

    print("Inserting low-freq words randomly...")
    np.random.seed(seed)
    for w, c in word2freq_low.items():
        print(f"Word {w}, count={c}")
        n = len(words)
        for _ in tqdm(range(c)):
            i = np.random.randint(n)
            words.insert(i, w)

    print("Writing new corpus inserting high-freq words")
    n = len(words)
    np.random.seed(seed)
    with open(outfile, mode="w", encoding="utf-8") as f:
        pbar = tqdm(total=n)
        while words:
            for w, freq in word2freq_high.items():
                if np.random.rand() < freq / n:
                    f.write(w + " ")
            w = words.pop(0)
            if not w.endswith('\n'):
                f.write(w + " ")
            else:
                f.write(w)
            pbar.update(1)
        pbar.close()

    # new_words = list()
    # for k, v in words2freq.items():
    #     new_words.extend([k] * v)
    #
    # print("Shuffling new words...")
    # np.random.seed(seed)
    # np.random.shuffle(new_words)
    #
    # print("Sampling indices of new words...")
    # np.random.seed(seed)
    # n = len(new_words)
    # n_tot = n + len(words)
    # # indices = np.random.choice(n_tot, n, replace=False)
    # new_indices = np.random.default_rng().choice(n_tot, n, replace=False)
    # new_indices.sort()
    # # other_indices = list(set(range(n_tot)) - set(new_indices))
    # # other_indices = np.array(other_indices)
    # # # other_indices = np.array([i for i in tqdm(range(n_tot)) if i not in new_indices])
    #
    # print("Creating new corpus...")
    # new_corpus = new_words
    # del new_words
    # step_size = len(words) // 100_000
    # for i in tqdm(range(100_000+1), desc="Adding actual words"):
    #     new_corpus += words[:step_size]
    #     del words[:step_size]
    # # new_corpus += words
    # # del words
    # # # new_corpus = list()
    # # # for i in tqdm(range(n)):
    # # #     new_corpus.append(new_words.pop(0))
    # # # for i in tqdm(range(n_tot - n)):
    # # #     new_corpus.append(words.pop(0))
    # new_corpus = np.array(new_corpus, dtype=object)
    # mask = np.full(n_tot, True)
    # mask[new_indices] = False
    # other_indices = np.arange(n_tot)
    # other_indices = other_indices[mask]
    # indices = np.concatenate([new_indices, other_indices])
    # del new_indices
    # del other_indices
    # indices = np.argsort(indices)
    # new_corpus = new_corpus[indices]
    # #
    # # new_corpus = np.full(n_tot, "", dtype=object)
    # # new_corpus[new_indices] = new_words
    # # del new_words
    # # del new_indices
    # # new_corpus[other_indices] = words
    # # del words
    # # del other_indices
    #
    # print("Writing new corpus word by word")
    # with open(outfile, mode="w", encoding="utf-8") as f:
    #     for w in tqdm(new_corpus):
    #         if w.endswith('\n'):
    #             f.write(w)
    #         else:
    #             f.write(w + " ")
    #
    # # print("Writing new corpus word by word")
    # # with open(outfile, mode="w", encoding="utf-8") as f:
    # #     for i in tqdm(range(n_tot)):
    # #         if i not in indices:
    # #             w = words.pop(0)
    # #             if w.endswith('\n'):
    # #                 f.write(w)
    # #             else:
    # #                 f.write(w + " ")
    # #         else:
    # #             _ = indices.pop(0)
    # #             w = new_words.pop(0)
    # #             f.write(w + " ")

    # np.savetxt(outfile, x, newline=" ", fmt="%s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('corpusfile', type=str)
    parser.add_argument('vocabfile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('seed', type=int)

    args = parser.parse_args()

    print("START -- ", datetime.datetime.now())
    main(args.corpusfile, args.vocabfile, args.outfile, args.seed)
    print("END -- ", datetime.datetime.now())
