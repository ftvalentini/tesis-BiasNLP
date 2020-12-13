
## Measuring stereotypes in corpora

A comparison between co-cooccurrence-based metrics and embedding-based metrics.

### Corpus setup

Follow these steps to create the data needed to run tests. Code was tested with:
* English wikipedia (`enwiki-20200701-pages-articles-multistream.xml.bz2` from https://dumps.wikimedia.org/simplewiki/)
* Simple English wikipedia (`simplewiki-20200620-pages-articles-multistream,xml.bz2` from https://dumps.wikimedia.org/simplewiki/)

**(1)** Build txt corpus out of wikipedia dump `.bz` file\
`python scripts/00-make_wiki_corpus.py <input_wiki_file> <output_file>`
* must determine `MAX_WC` `ARTICLE_MIN_WC` `ARTICLE_MAX_WC` in script
* outputs a `.txt` corpus, a `.meta.txt` with corpus info and `.meta.json` with documents' info
* `<output_file>` must not have extension.

### GloVe

The steps to create a GloVe word embeddings numpy array from the Wikipedia corpus are:

**0.** Build GloVe from source only once so that it can be executed\

In Windows:
```
make -C "GloVe"
```

In Linux:
```
cd GloVe && make
```

**1.** Train word embeddings and save them as `.bin`

* set `DISTANCE_WEIGHTING=1` in `glove.config` so that co-occurrence counts are normalized as done in vanilla GloVe
* set `VOCAB_MIN_COUNT=20` so that words are counted as in vanilla GloVe (words with frequency less than 20 are removed _before_ windows of `WINDOW_SIZE`)
* `CORPUS_ID` is defined according to order in `CORPORA` list in `scripts/corpora_dict`

```
rm nohup.out
chmod +x scripts/01-glove.sh
nohup scripts/01-glove.sh scripts/glove.config &
```

**2.** Extract word vectors matrices

Read GloVe `.bin` and saves all vectors (target vectors, context vectors and both biases) as a tuple in a `glove_matrices-<>.pkl` file. Besides, save a `glove-<...>.npy` with an array containing a vector for each word, such that each vector is the sum of the word's target and context vectors.

Example:
```
VOCABFILE="embeddings/vocab-C3-V20.txt"
BINFILE="embeddings/vectors-C3-V20-W5-D1-D100-R0.05-E50-S1.bin"
MATFILE="embeddings/full_vectors-C3-V20-W5-D1-D100-R0.05-E50-S1.pkl"
EMBEDFILE="embeddings/glove-C3-V20-W5-D1-D100-R0.05-E50-S1.npy"
python3 -u scripts/02-build_glove_matrices.py $VOCABFILE $BINFILE $MATFILE $EMBEDFILE
```

### Word2Vec

In order to train word embeddings with SGNS or CBOW using python `gensim` with parameters similar to the ones used for GloVe:

**1.** Save `.model` with trained model and `.npy` with embeddings in array format

If model is large, files with extension `.trainables.syn1neg.npy` and `.wv.vectors.npy` might be saved alongside `.model`.

Example:
```
CORPUSID=3
VOCABFILE="embeddings/vocab-C3-V20.txt"
CORPUSFILE="corpora/enwikiselect.txt"
OUTDIR="E:\\tesis-BiasNLP"
SG=1 # 0:cbow, 1:sgns
SIZE=100
WINDOW=5
MINCOUNT=20
SEED=1
python -u scripts/02-train_word2vec.py \
  --id $CORPUSID --corpus $CORPUSFILE --vocab $VOCABFILE --outdir $OUTDIR \
  --size $SIZE --window $WINDOW --count $MINCOUNT --sg $SG --seed $SEED
```

### Co-occurrence and PMI matrices

Follow these steps to create co-occurrence and PMI (Point Mutual Information) sparse matrices from the output of GloVe:

**1.** Build a `.bin` with cooccurrence counts and `.txt` with vocabulary

* must specify `DISTANCE_WEIGHTING=0` in `cooc.config` so that co-occurrence counts are raw (not weighted by distance to center)
* if `VOCAB_MIN_COUNT=1` then all words are kept when computing co-occurrences -- otherwise use the same `VOCAB_MIN_COUNT` as in GloVe or word2vec
* `CORPUS_ID` is defined according to order in `CORPORA` list in `scripts/corpora_dict`

```
chmod +x scripts/01-cooc.sh
nohup scripts/01-cooc.sh scripts/cooc.config &
```

**2.** Save scipy.sparse co-occurrence matrix as `.npz` file

Co-occurrences stored in `.bin` file are used as input

Example:
```
rm nohup.out
VOCABFILE="embeddings/vocab-C3-V20.txt"
COOCFILE="embeddings/cooc-C3-V20-W5-D0.bin"
OUTFILE="embeddings/cooc-C3-V20-W5-D0.npz"
nohup python3 -u scripts/02-build_cooc_matrix.py \
  -v $VOCABFILE -c $COOCFILE -o $OUTFILE &
```

**3.** Save scipy.sparse PMI matrix as `.npz` file

The input is the `.npz` co-occurrence matrix

Example:
```
rm nohup.out
MATRIX="embeddings/cooc-C3-V20-W5-D0.npz"
OUTFILE="embeddings/pmi-C3-V20-W5-D0.npz"
nohup python3 -u scripts/03-build_pmi_matrix.py \
  --matrix $MATRIX --outfile $OUTFILE &
```

### Bias by word

Get the value of bias of each word (w) in the vocabulary with respect to given sets of target words (a & b). Bias for each word w is measured in four ways:

* DPPMI bias (Difference of Positive Mointwise Mutual Information): PPMI(w,a) - PPMI(w,b)
* PPMI_vec bias (difference of cosines of DPPMI vectors): cosine(w,a) - cosine(w,b)
* word2vec bias (difference of cosines of SGNS vectors): cosine(w,a) - cosine(w,b)
* GloVe bias (difference of cosines of GloVe vectors): cosine(w,a) - cosine(w,b)

Additionally, other quantities are measured and saved, such as the norm of each word's vector, the approximation of PMI by log odds-ratio, etc.

The lists of target words to be used must be saved in `words_lists/`.

#### Co-occurrence based biases

Compute DPPMI and PPMI_vec biases of each word in vocabulary with respect to a given set of target groups (for example, `HE` and `SHE`). Results are saved as `.csv` in `results/`.

Example:
```
VOCABFILE="embeddings/vocab-C3-V20.txt"
COOCFILE="embeddings/cooc-C3-V20-W5-D0.npz"
PMIFILE="embeddings/pmi-C3-V20-W5-D0.npz"
TARGETA="HE"
TARGETB="SHE"
python3 -u scripts/04-biasbyword_cooc.py \
  --vocab $VOCABFILE --cooc $COOCFILE --pmi $PMIFILE --a $TARGETA --b $TARGETB
```

#### Word embeddings based biases

Get the word2vec and GloVe biases of each word in vocabulary with respect to a given set of target groups (for example, `HE` and `SHE`). Results are saved as `.csv` in `results/`.

Example:
```
VOCABFILE="embeddings/vocab-C3-V20.txt"
TARGETA="HE"
TARGETB="SHE"

EMBEDFILE="embeddings/w2v-C3-V20-W5-D100-SG1-S1.npy"
python -u scripts/04-biasbyword_we.py \
  --vocab $VOCABFILE --matrix $EMBEDFILE --a $TARGETA --b $TARGETB

EMBEDFILE="embeddings/glove-C3-V20-W5-D1-D100-R0.05-E50-S1.npy"
python -u scripts/04-biasbyword_we.py \
  --vocab $VOCABFILE --matrix $EMBEDFILE --a $TARGETA --b $TARGETB
```

#### Join results

Join all csvs with results of biases by word to make one csv with all the relevant data, which is saved in `results/`.

Example:
```
ID=3
TARGETA="HE"
TARGETB="SHE"
python -u scripts/05-biasbyword_join.py --corpus $ID --a $TARGETA --b $TARGETB
```

### Insertion of random ficticious words

In order to assess the influence of word frequency on word embedding bias, we insert new words randomly on the original corpus. We add 6 different context words with frequencies of varying orders of magnitude, and 2 target words with frequencies similar to those of "he" and "she". We run this five times with different random seeds, so that as a result we get five different corpora with words inserted randomly.

#### Create new corpora

**1.** Hardcode the name of the lists of the target words in `scripts/insert_new_words.py`.

**2.** Hardcode the name of the input corpus file, the name of the vocabulary file and the list of random seeds in `scripts/insert_newwords_multiple.sh`.

New corpora are saved with the format `corpora/<name>_newwords_S<seed>.txt`.

Example:
```
rm nohup.out
chmod +x scripts/insert_newwords_multiple.sh
nohup bash scripts/insert_newwords_multiple.sh &
```

#### Train multiple embeddings

Train GloVe and word2vec once for each of the five perturbed corpora. GloVe must be trained first because its vocabulary is used in a check when running word2vec.

**1.** Write the names of the perturbed copora in `corpora_dict`.

**2.** Hardcode the IDs of the perturbed copora (order in `corpora_dict`) in `scripts/train_multiple_glove.sh` and `scripts/train_multiple_w2v.sh`.

**3.** Train GloVe with the same set-up (`scripts/glove.config` file) used in [GloVe](#glove) and save GloVe embeddings.

Example:
```
rm nohup.out
chmod +x scripts/01-glove.sh
chmod +x scripts/train_multiple_glove.sh
nohup bash scripts/train_multiple_glove.sh &
```

**4.** Hardcode the configuration of word2vec in `scripts/train_multiple_w2v.sh` -- use the same configuration used in [Word2Vec](#word2vec).

Example:
```
rm nohup.out
chmod +x scripts/train_multiple_w2v.sh
nohup bash scripts/train_multiple_w2v.sh &
```

#### Create multiple co-occurrence and PMI matrices

**1.** Hardcode the IDs of the perturbed copora (order in `corpora_dict`) in `scripts/train_multiple_cooc.sh`.

**2.** Hardcode the same set-up used in [Co-occurrence and PMI matrices](#co-occurrence-and-pmi-matrices) in `scripts/cooc.config` file in order to build co-occurrence and PMI matrices for each of the perturbed corpora.

Example:
```
rm nohup.out
chmod +x scripts/train_multiple_cooc.sh
nohup bash scripts/train_multiple_cooc.sh &
```

#### Compute biases

Compute DPPMI, PPMI_vec, GloVe and word2vec biases of each word of each corpus with respect to the sets of ficticious targete words (T1,T2) and/or other sets. Results are saved as `.csv` in `results/`.

Example:
```
rm nohup.out
TARGET_A="HE"
TARGET_B="SHE"
chmod +x scripts/biasbyword_multiple.sh
nohup bash scripts/biasbyword_multiple.sh $TARGET_A $TARGET_B &
rm nohup.out
TARGET_A="T1"
TARGET_B="T2"
chmod +x scripts/biasbyword_multiple.sh
nohup bash scripts/biasbyword_multiple.sh $TARGET_A $TARGET_B &
```



<!-- FALTA:
- test undersampled/oversampled_corpora
### Influence of frequency in bias metrics
Create new perturbed corpora where the ratio of female/male pronouns ("he"/"she") is altered. Assess the impact on the relationship between word frequency and gender bias as measured.
**1.** Count number of target pronouns in each document of the corpus\
`python scripts/count_target_words.py <corpus_txt> <word_a> <word_b>`
Example:
```
CORPUS="corpora/enwikiselect.txt"
A="he"
B="she"
python scripts/count_words.py $CORPUS $A $B
```
**2.** Create over and undersampled corpora\
`python -u scripts/make_undersampled_corpora.py <corpus.txt> <vocab.txt> <counts.pkl> <output_dir> <seed>`
`python -u scripts/make_oversampled_corpora.py <corpus.txt> <vocab.txt> <counts.pkl> <output_dir> <seed>`
* must set `WORD_A`, `WORD_B`, `RATIOS` in scripts
Example:
```
CORPUSFILE="corpora/enwikiselect.txt"
VOCABFILE="embeddings/vocab-C3-V1.txt"
COUNTSFILE="corpora/enwikiselect_counts_he-she.pkl"
OUTDIR="E:\\tesis-BiasNLP\\corpora"
SEED=88
python -u scripts/make_undersampled_corpora.py $CORPUSFILE $VOCABFILE $COUNTSFILE $OUTDIR $SEED
python -u scripts/make_oversampled_corpora.py $CORPUSFILE $VOCABFILE $COUNTSFILE $OUTDIR $SEED
```
**3.** (...)
rm nohup.out
chmod +x scripts/01-embed.sh
chmod +x scripts/03-train_multiple_glove.sh
nohup scripts/03-train_multiple_glove.sh &
tail -f nohup.out
chmod +x scripts/03-train_multiple_w2v.sh
rm nohup.out
nohup bash scripts/03-train_multiple_w2v.sh &
tail -f nohup.out -->

<!-- ULTIMO PONER: scripts que hacen plots
(una seccion para cada tipo de plot)
```
python -u scripts/plots_frequency.py
```
### Stopwords and frequency analysis

Analyse visually the relationship between:
a) bias of each word as measured by DPMI and RND with respect to predefined sets of target words (for example, `MALE` and `FEMALE`)
b) stopwords and word frequency

**(1)** Run `test_distbyword.py <vocab_file> <embed_file>` and `test_dpmibyword.py`
* `TARGET_A`,`TARGET_B` are names of target word lists in `words_lists/`

**(2)** Get plots for targets set in (1)\
`python -u test_stopwords.py <dpmi_file> <we_file>`
* plots are saved in `results/plots/`

Example:
```
FILE_DPMI="results/csv/dpmibyword_C3_MALE_SHORT-FEMALE_SHORT.csv"
FILE_WE="results/csv/distbyword_glove-C3_MALE_SHORT-FEMALE_SHORT.csv"
#FILE_WE="results/csv/distbyword_w2v-C3_MALE_SHORT-FEMALE_SHORT.csv"
python -u test_stopwords.py $FILE_DPMI $FILE_WE
```

**(3)** Get plots for arbitrary target words\
`python test_frequency.py <vocab_file> <embed_file>`
* plots are saved in `results/plots/`

Example:
```
VOCABFILE="embeddings/vocab-C3-V20.txt"
EMBEDFILE="embeddings/glove-C3-V20-W8-D1-D100-R0.05-E150-S1.npy"
# EMBEDFILE="embeddings/w2v-C3-V20-W8-D100-SG0.npy"
python -u test_frequency.py $VOCABFILE $EMBEDFILE
``` -->



<!--
CORPUSID=9
VOCABFILE="embeddings/vocab-C9-V20.txt"
CORPUSFILE="corpora/enwikiselect_newwords.txt"
OUTDIR=""
SG=1
SIZE=100
WINDOW=8
MINCOUNT=20
SEED=1
nohup python3 -u scripts/02-train_word2vec.py \
  --id $CORPUSID --corpus $CORPUSFILE --vocab $VOCABFILE --outdir $OUTDIR \
  --size $SIZE --window $WINDOW --count $MINCOUNT --sg $SG --seed $SEED &

VOCABFILE="embeddings/vocab-C9-V20.txt"
EMBEDFILE="embeddings/vectors-C9-V20-W8-D1-D100-R0.05-E150-S1.bin"
OUTFILE="embeddings/glove-C9-V20-W8-D1-D100-R0.05-E150-S1.npy"
FILE_W2V="embeddings/w2v-C9-V20-W8-D100-SG1.npy"
FILE_GLOVE="embeddings/glove-C9-V20-W8-D1-D100-R0.05-E150-S1.npy"
# glove matrices
python3 -u scripts/02-build_glove_matrix.py -v $VOCABFILE -e $EMBEDFILE -o $OUTFILE
TARGETA="T1"
TARGETB="T2"
# w2v
python3 -u scripts/04-biasbyword_we.py \
  --vocab $VOCABFILE --matrix $FILE_W2V --a $TARGETA --b $TARGETB
# glove
python3 -u scripts/04-biasbyword_we.py \
  --vocab $VOCABFILE --matrix $FILE_GLOVE --a $TARGETA --b $TARGETB

## cooc
#python3 -u scripts/04-biasbyword_cooc.py \
#  --vocab $VOCABFILE --matrix $COOCFILE --a $TARGETA --b $TARGETB

-->


### Important references
- [Garg et al 2018 -- GitHub repo](https://github.com/nikhgarg/EmbeddingDynamicStereotypes)
- [Brunet et al 2019 -- GitHub repo](https://github.com/mebrunet/understanding-bias)



<!-- ## Backlog

### Bias gradient in GloVe

**(1)** Extract all GloVe vectors\
`python scripts/02-build_glove_full_vectors.py <vocab_file.txt> <embed_file.bin> <out_file.pkl>`
* Reads `.bin` and saves all vectors (word vectors, context vectors and both biases) as a tuple in a `.pkl`

Example:
```
cd tesis-BiasNLP
VOCABFILE="embeddings/vocab-C3-V20.txt"
EMBEDFILE="embeddings/vectors-C3-V20-W8-D1-D100-R0.05-E150-S1.bin"
OUTFILE="embeddings/full_vectors-C3-V20-W8-D1-D100-R0.05-E150-S1.pkl"
python scripts/build_glove_full_vectors.py $VOCABFILE $EMBEDFILE $OUTFILE
```

**(2)** Build scipy.sparse co-occurrence matrix\
`python scripts/02-build_cooc_matrix.py <vocab_file.txt> <cooc_file.bin> <out_file.npz>`
* builds a `scipy.sparse.csr_matrix` with co-occurrences stored in `.bin` file
* the `.bin` file should be the same one used to train GloVe
* matrix is saved as `.npz` file

Example:
```
cd tesis-BiasNLP
VOCABFILE="embeddings/vocab-C3-V20.txt"
COOCFILE="embeddings/cooc-C3-V20-W8-D1.bin"
OUTFILE="embeddings/cooc-C3-V20-W8-D1.npz"
nohup python3 scripts/02-build_cooc_matrix.py -v $VOCABFILE -c $COOCFILE -o $OUTFILE 1>test.out 2>test.err &
tail -f test.err
``` -->



<!-- ## OLD STUFF

### Relative norm distance (RND) bias

Get the value of RND bias in corpus for given sets of target and context words (for example, `MALE`, `FEMALE` and `SCIENCE`)

**(1)** Set parameters in `test_rnd.py`
* `TARGET_A`,`TARGET_B`,`CONTEXT` are names of word lists in `words_lists/`

**(2)** Get value of relative norm distance\
`python test_rnd.py`
* Results are saved as `.md` in `results/`

### Differential PMI bias (DPMI)

Get the value of DPMI bias in corpus for given sets of target and context words (for example, `MALE`, `FEMALE` and `SCIENCE`)

**(1)** Set parameters in `test_dpmi.py`
* `TARGET_A`,`TARGET_B`,`CONTEXT` are names of word lists in `words_lists/`

**(2)** Get values of diff. PMI and log-OddsRatio approximation\
`python test_dpmi.py`
* Results are saved as `.md` in `results/`


 -->
