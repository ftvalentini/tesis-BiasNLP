STEPS:

0. `make -C "GloVe"`: run once and for all to build GloVe from source so that it can be executed

1. `python scripts/00-make_wiki_corpus.py`: makes txt corpus out of wikipedia dump `.bz` file in `/corpora` -- must determine `MAX_WC` `ARTICLE_MIN_WC` `ARTICLE_MAX_WC` in script. Outputs a .txt corpus, a .meta.txt with corpus info and .meta.json with documents' info.
2. `scripts/01-cooc.sh scripts/cooc.config`: builds .bin with coocurrence counts and .txt with vocabulary. Must specify `DISTANCE_WEIGHTING=0` in .config so that cooc. counts are raw.
3. `scripts/01-embed.sh scripts/glove.config`: trains GloVe and saves them in .bin. Set `DISTANCE_WEIGHTING=1` in .config so that cooc. counts are normalized as done in vanilla GloVe.
4. run some tests in `/tests` and print results to some `.md`

TODO:
- check que cooc dict siempre tenga (i,j) y (j,i)
- validar bootrstap interval de Garg
- 'chef' not in vocab --> armar dataset temporal de enwiki con https://archive.org/search.php?query=subject%3A%22enwiki%22+AND+subject%3A%22data+dumps%22+AND+collection%3A%22wikimediadownloads%22&page=2

Notes:
- Cooc. de GloVe solo considera words in vocab (>VOCAB_MIN_COUNT) como parte de la ventana -- es decir si window_size=2 y una palabra contexto está a 5 palabras de distancia de la target pero ninguna de estas está en el vocab, entonces es como si la distancia fuera 1 y entonces está dentro de la ventana. Entonces cooccurence Glove y `utils.coocurrence.create_cooc_dict()` coinciden solo si `VOCAB_MIN_COUNT=1`.
- en step 1 usé Google Colab (ver https://colab.research.google.com/drive/143Me55jclH1DFEzUsFnHB0liq1fsuszr?authuser=1 con rkf.valentini@gmail.com)
- Glove pondera coocurrencias por distancia (+ distancia en la ventana --> - coocucrrencia) -- esto se desactiva con `DISTANCE_WEIGHTING=0`
- cooc.sh deletes stuff from embed.sh related to vectors and sets DISTANCE_WEIGHTING to 0.
- had to install gcc to make GloVe -- used:
  - https://stackoverflow.com/questions/25705726/bash-gcc-command-not-found-using-cygwin-when-compiling-c
  - https://cygwin.com/install.html
- downloaded `simplewiki-20200620-pages-articles-multistream.xml.bz2` from https://dumps.wikimedia.org/simplewiki/


Code references:
- [Garg et al 2018](https://github.com/nikhgarg/EmbeddingDynamicStereotypes)
- [Brunet et al 2019](https://github.com/mebrunet/understanding-bias)
