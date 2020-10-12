
## Measuring stereotypes in corpora

A comparison between co-coocurrence-based metrics and embedding-based metrics.

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

**(0)** Build GloVe from source so that it can be executed\
`make -C "GloVe"` (run only once)

**(1)** Train GloVe
`scripts/01-embed.sh scripts/glove.config`\
* trains word embeddings and saves them in `.bin`
* set `DISTANCE_WEIGHTING=1` in `glove.config` so that co-ocurrence counts are normalized as done in vanilla GloVe
* set `VOCAB_MIN_COUNT=20` so that words are counted as in vanilla GloVe (words with frequency less than 20 are removed and not considered when building windows of `WINDOW_SIZE`)
* `CORPUS_ID` is defined according to order in `CORPORA` list in `scripts/01-embed.sh`

**(2)** Extract GloVe word vectors matrix\
`python scripts/02-build_embeddings_matrix.py <vocab_file.txt> <embed_file.bin> <out_file.npy>`
* Reads `.bin` and saves all vectors (word vectors, context vectors and both biases) as a tuple in a `.npy` file

Example:
```
cd tesis-BiasNLP
VOCABFILE="embeddings/vocab-C3-V20.txt"
EMBEDFILE="embeddings/vectors-C3-V20-W8-D1-D100-R0.05-E150-S1.bin"
OUTFILE="embeddings/glove-C3-V20-W8-D1-D100-R0.05-E150-S1.npy"
python3 scripts/02-build_embeddings_matrix.py -v $VOCABFILE -e $EMBEDFILE -o $OUTFILE
```

### Word2Vec

Train word embeddings with Skip-Gram or CBOW using `gensim` with parameters similar to the ones used for GloVe.

**(1)** Train Word2Vec\
`python -u scripts/train_word2vec.py <required_params>`
* Saves `.model` with trained model and `.npy` with embeddings in array format.
* If model is large, files with extension `.trainables.syn1neg.npy` and `.wv.vectors.npy` might be saved alongside `.model`.

Example:
```
CORPUSID=3
VOCABFILE="embeddings/vocab-C3-V20.txt"
CORPUSFILE="corpora/enwikiselect.txt"
OUTDIR="E:\\tesis-BiasNLP"
SG=0 # 0:cbow, 1:sgns
SIZE=100
WINDOW=8
MINCOUNT=20
SEED=1
python -u scripts/train_word2vec.py \
  --id $CORPUSID --corpus $CORPUSFILE --vocab $VOCABFILE --outdir $OUTDIR \
  --size $SIZE --window $WINDOW --count $MINCOUNT --sg $SG --seed $SEED
```

### Co-ocurrence matrix

**(1)** Build raw co-ocurrence matrix and vocabulary\
`scripts/01-cooc.sh scripts/cooc.config <corpus_dir> <results_dir>`
* builds `.bin` with coocurrence counts and `.txt` with vocabulary
* must specify `DISTANCE_WEIGHTING=0` in `cooc.config` so that co-ocurrence counts are raw (not weighted by distance to center)
* if `VOCAB_MIN_COUNT=1` then all words are part of window.
* `CORPUS_ID` is defined according to order in `CORPORA` list in `scripts/01-cooc.sh`

**(2)** Build scipy.sparse co-ocurrence matrix\
`python scripts/02-build_cooc_matrix.py <vocab_file.txt> <cooc_file.bin> <out_file.npz>`
* builds a `scipy.sparse.csr_matrix` with co-occurrences stored in `.bin` file
* matrix is saved as `.npz` file

Example:
```
cd tesis-BiasNLP
VOCABFILE="embeddings/vocab-C3-V1.txt"
COOCFILE="embeddings/cooc-C3-V1-W8-D0.bin"
OUTFILE="embeddings/cooc-C3-V1-W8-D0.npz"
nohup python3 scripts/02-build_cooc_matrix.py -v $VOCABFILE -c $COOCFILE -o $OUTFILE 1>test.out 2>test.err &
tail -f test.err
```

### Relative norm distance (RND) bias

Get the value of RND bias in corpus for given sets of target and context words (for example, `MALE`, `FEMALE` and `SCIENCE`)

**(1)** Set parameters in `test_rnd.py`
* `TARGET_A`,`TARGET_B`,`CONTEXT` are names of word lists in `words_lists/`

**(2)** Get value of relative norm distance\
`python test_rnd.py`
* Results are saved as `.md` in `results/`

<!-- VER DESDE aca para ABAJO: -->
  <!-- SE TIENE QUE PODER CAMBIAR LOS EMBEDDINGS COMO PARAM PARA USAR W2V -->


### RND and relative cosine similarity (RCS) by word

Get the RND and RCS bias of each word in vocabulary with respect to a given set of target groups (for example, `MALE` and `FEMALE`) and a set of trained vectors.

**(1)** Set parameters in `test_distbyword.py`
* `TARGET_A`,`TARGET_B` are names of target word lists in `words_lists/`

**(2)** Get value of relative norm distance for each word with respect to the two groups of target words\
`python test_distbyword.py <vocab_file> <embed_file>`
* Results are saved as `.md` and `.csv` in `results/`

Example:
```
VOCABFILE="embeddings/vocab-C3-V20.txt"
# EMBEDFILE="embeddings/glove-C3-V20-W8-D1-D100-R0.05-E150-S1.npy"
EMBEDFILE="embeddings/w2v-C3-V20-W8-D100-SG0.npy"
python -u test_distbyword.py $VOCABFILE $EMBEDFILE
```

### Differential PMI bias (DPMI)

Get the value of DPMI bias in corpus for given sets of target and context words (for example, `MALE`, `FEMALE` and `SCIENCE`)

**(1)** Set parameters in `test_dpmi.py`
* `TARGET_A`,`TARGET_B`,`CONTEXT` are names of word lists in `words_lists/`

**(2)** Get values of diff. PMI and log-OddsRatio approximation\
`python test_dpmi.py`
* Results are saved as `.md` in `results/`

### DPMI bias by word

Get the DPMI bias of each word in vocabulary with respect to a given set of target groups (for example, `MALE` and `FEMALE`)

**(1)** Set parameters in `test_dpmibyword.py`
* `TARGET_A`,`TARGET_B` are names of target word lists in `words_lists/`

**(2)** Get values of DPMI and log-OddsRatio for each word with respect to the two groups of target words\
`python test_dpmibyword.py`
* Results are saved as `.md` and `.csv` in `results/`

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
#FILE_WE="results/csv/distbyword_glove-C3_MALE_SHORT-FEMALE_SHORT.csv"
FILE_WE="results/csv/distbyword_w2v-C3_MALE_SHORT-FEMALE_SHORT.csv"
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
```

### Influence of frequency in bias metrics

Create new perturbed corpora where the ratio of female/male pronouns ("he"/"she") is altered. Assess the impact on the relationship between word frequency and gender bias as measured.

**(1)** Count number of target pronouns in each document of the corpus\
`python scripts/count_target_words.py <corpus_txt> <word_a> <word_b>`

Example:
```
cd tesis-BiasNLP
CORPUS="corpora/enwikiselect.txt"
A="he"
B="she"
python scripts/count_words.py $CORPUS $A $B
```

**(2)** Create over and undersampled corpora\
`python -u scripts/make_undersampled_corpora.py <corpus.txt> <vocab.txt> <counts.pkl> <output_dir> <seed>`
`python -u scripts/make_oversampled_corpora.py <corpus.txt> <vocab.txt> <counts.pkl> <output_dir> <seed>`
* must set `WORD_A`, `WORD_B`, `RATIOS` in scripts

Example:
```
cd tesis-BiasNLP
CORPUSFILE="corpora/enwikiselect.txt"
VOCABFILE="embeddings/vocab-C3-V1.txt"
COUNTSFILE="corpora/enwikiselect_counts_he-she.pkl"
OUTDIR="E:\\tesis-BiasNLP\\corpora"
SEED=88
python -u scripts/make_undersampled_corpora.py $CORPUSFILE $VOCABFILE $COUNTSFILE $OUTDIR $SEED
python -u scripts/make_oversampled_corpora.py $CORPUSFILE $VOCABFILE $COUNTSFILE $OUTDIR $SEED
```

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

**(2)** Build scipy.sparse co-ocurrence matrix\
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
```

### Important references
- [Garg et al 2018 -- GitHub repo](https://github.com/nikhgarg/EmbeddingDynamicStereotypes)
- [Brunet et al 2019 -- GitHub repo](https://github.com/mebrunet/understanding-bias)
