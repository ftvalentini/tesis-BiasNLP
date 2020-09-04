
## Measuring stereotypes in corpora

### Comparison between co-coocurrence-based metrics and embedding-based metrics

Follow these steps to create the data needed to run tests. Code was tested with English and Simple English wikipedias (`simplewiki-20200620-pages-articles-multistream,xml.bz2` from https://dumps.wikimedia.org/simplewiki/ and `enwiki-20200701-pages-articles-multistream.xml.bz2` from https://dumps.wikimedia.org/simplewiki/).

0. Build GloVe from source so that it can be executed
`make -C "GloVe"` (run only once)

1. Build txt corpus out of wikipedia dump `.bz` file
`python scripts/00-make_wiki_corpus.py <input_wiki_file> <output_file>`: outputs a .txt corpus, a .meta.txt with corpus info and .meta.json with documents' info. Must determine `MAX_WC` `ARTICLE_MIN_WC` `ARTICLE_MAX_WC` in script `<output_file>` must not have extension.

2. Build co-ocurrence matrix and vocabulary
`scripts/01-cooc.sh scripts/cooc.config <corpus_dir> <results_dir>`: builds `.bin` with coocurrence counts and `.txt` with vocabulary. Must specify `DISTANCE_WEIGHTING=0` in `cooc.config` so that co-ocurrence counts are raw. If `VOCAB_MIN_COUNT=1` then all words are part of window.

3. Train GloVe
`scripts/01-embed.sh scripts/glove.config`: trains word embeddings and saves them in `.bin`. Set `DISTANCE_WEIGHTING=1` in `glove.config` so that co-ocurrence counts are normalized as done in vanilla GloVe.

<!-- ** poner la creacion de matriz de embeddings **
** poner las corridas de PMI **
** poner los tests ** -->
<!-- run tests with relative norm distance bias (Garg et al 2018): -->

### Bias gradient in GloVe

1. Save all GloVe vectors (word vectors, context vectors and both biases) as tuple in `.pkl`
`python scripts/build_glove_full_vectors.py <vocab_file.txt> <embed_file.bin> <out_file.pkl>`



**TODO**
- Pensar: los resultados de glove vs pmi pueden tener que ver con que GloVe mide cooc excluyendo a las palabras out of vocab? (es decir, elimina estas palabras y LUEGO mide coocurrencias en ventanas de tamaño window_size que solo tienen a las palabras del vocab). Además GloVe usa distance weighting.


Notes:
- Cooc. de GloVe solo considera words in vocab (>VOCAB_MIN_COUNT) como parte de la ventana -- es decir si window_size=2 y una palabra contexto está a 5 palabras de distancia de la target pero ninguna de estas está en el vocab, entonces es como si la distancia fuera 1 y entonces está dentro de la ventana. Entonces cooccurence Glove y `utils.coocurrence.create_cooc_dict()` coinciden solo si `VOCAB_MIN_COUNT=1`.
- Cooc. de Glove cuenta todas las apariciones del contexto en la ventana (ej. si en una ventana el contexto está tres veces, se cuenta las 3) --> entonces la suma de las coocurrencias para una palabra con el resto no es igual a la frecuencia de la palabra
- en step 1 usé Google Colab (ver https://colab.research.google.com/drive/143Me55jclH1DFEzUsFnHB0liq1fsuszr?authuser=1 con rkf.valentini@gmail.com)
- Glove pondera coocurrencias por distancia (+ distancia en la ventana --> - coocucrrencia) -- esto se desactiva con `DISTANCE_WEIGHTING=0`
- cooc.sh deletes stuff from embed.sh related to vectors and sets DISTANCE_WEIGHTING to 0.
- had to install gcc to make GloVe -- used:
  - https://stackoverflow.com/questions/25705726/bash-gcc-command-not-found-using-cygwin-when-compiling-c
  - https://cygwin.com/install.html
- downloaded `simplewiki-20200620-pages-articles-multistream.xml.bz2` from https://dumps.wikimedia.org/simplewiki/
- out of vocab: "sunnites" "peeved"


Code references:
- [Garg et al 2018](https://github.com/nikhgarg/EmbeddingDynamicStereotypes)
- [Brunet et al 2019](https://github.com/mebrunet/understanding-bias)
