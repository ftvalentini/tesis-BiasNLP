import numpy as np
import argparse
from tqdm import tqdm

from scripts.utils.corpora import load_vocab


TARGET_A = "HE"
TARGET_B = "SHE"


def main(corpusfile, vocabfile, outfile, seed):

    str2idx, idx2str, str2count = load_vocab(vocabfile)
    # new words with frecuencies (t1, t2, c1, c2, ..., c6)
    words2freq = {'c'+str(i): 2*(10**i) for i in range(1,7)}
    words2freq['t1'] = str2count[TARGET_A.lower()]
    words2freq['t2'] = str2count[TARGET_B.lower()]

    print("Reading corpus...")
    # read all words at once (including newlines)
    with open(corpusfile, encoding="utf-8") as f:
        words = f.read().split(" ")

    print("Inserting words at random...")
    np.random.seed(seed)
    for w, c in words2freq.items():
        print(f"Word {w}, count={c}")
        n = len(words)
        indices = np.random.choice(n, size=c, replace=False)
        for i in tqdm(indices):
            words.insert(i, w)

    print("Creating new corpus")
    out = " ".join(words)

    print("Writing new corpus")
    with open(outfile, mode="w", encoding="utf-8") as f:
        f.writelines(out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('corpusfile', type=str)
    parser.add_argument('vocabfile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('seed', type=int)

    args = parser.parse_args()

    main(args.corpusfile, args.vocabfile, args.outfile, args.seed)
