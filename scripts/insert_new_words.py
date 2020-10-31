import numpy as np
import datetime
import argparse

from tqdm import tqdm
from blist import blist

from utils.corpora import load_vocab


TARGET_A = "HE"
TARGET_B = "SHE"

# corpusfile = "../corpora/test2.txt"

def main(corpusfile, vocabfile, outfile, seed):

    str2idx, idx2str, str2count = load_vocab(vocabfile)
    # new words with frecuencies (t1, t2, c1, c2, ..., c6)
    words2freq = {'c'+str(i): 2*(10**i) for i in range(1,7)}
    words2freq['t1'] = str2count[TARGET_A.lower()]
    words2freq['t2'] = str2count[TARGET_B.lower()]

    print("Reading and splitting corpus...")
    n_docs = sum(1 for line in open(corpusfile, 'r', encoding="utf-8"))
    words = blist() # blist tiene metodo .insert fast
    # words = list()
    with open(corpusfile, encoding="utf-8") as f:
        for doc in tqdm(f, total=n_docs):
            words_doc = doc.split(" ")
            words.extend(words_doc)

    # print("Reading corpus...")
    # # read all words at once (including newlines)
    # with open(corpusfile, encoding="utf-8") as f:
    #     # words = f.read().split(" ")
    #     docs = f.readlines()
    #
    # print("Splitting corpus in words...")
    # words = list()
    # for doc in tqdm(range(len(docs))):
    #     doc_i = docs.pop(0)
    #     words_i = doc_i.split(" ")
    #     words.extend(words_i)

    print("Inserting words randomly...")
    # words = blist(words)
    np.random.seed(seed)
    for w, c in words2freq.items():
        print(f"Word {w}, count={c}")
        n = len(words)
        for _ in tqdm(range(c)):
            i = np.random.randint(n)
            words.insert(i, w)

    # print("Inserting words at random...")
    # np.random.seed(seed)
    # for w, c in words2freq.items():
    #     print(f"Word {w}, count={c}")
    #     n = len(words)
    #     indices = np.random.choice(n, size=c, replace=False)
    #     for i in tqdm(indices):
    #         words.insert(i, w)

    print("Writing new corpus word by word")
    with open(outfile, mode="w", encoding="utf-8") as f:
        for i in tqdm(range(len(words))):
            w = words.pop(0)
            if w.endswith('\n'):
                f.write(w)
            else:
                f.write(w + " ")

    # print("Creating new corpus")
    # out = " ".join(words)
    #
    # print("Writing new corpus")
    # with open(outfile, mode="w", encoding="utf-8") as f:
    #     f.writelines(out)


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
