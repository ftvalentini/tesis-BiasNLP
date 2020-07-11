STEPS:

0. `make -C "GloVe"`: run once and for all to build GloVe from source so that it can be executed

1. `python scripts/00-make_wiki_corpus.py`: makes txt corpus out of wikipedia dump `.bz` file in `/corpora` -- must determine `MAX_WC` `ARTICLE_MIN_WC` `ARTICLE_MAX_WC` in script. Outputs a .txt corpus, a .meta.txt with corpus info and .meta.json with documents' info.
2. `scripts/01-cooc.sh scripts/cooc.config`: builds .bin with coocurrence counts and .txt with vocabulary. Must specify `DISTANCE_WEIGHTING=0` in .config so that cooc. counts are raw.
3. `scripts/01-embed.sh scripts/glove.config`: trains GloVe and saves them in .bin. Set `DISTANCE_WEIGHTING=1` in .config so that cooc. counts are normalized as done in vanilla GloVe.
4. run some tests in `/tests` and print results to some `.md`

TODO:
- cooccurence Glove y `utils.coocurrence.create_cooc_dict()` coinciden solo si `VOCAB_MIN_COUNT=1` -> revisar esto
- check que cooc dict siempre tenga (i,j) y (j,i)
- hacer bootrstap interval de Garg

Notes:
- en step 1 usé Google Colab (ver https://colab.research.google.com/drive/143Me55jclH1DFEzUsFnHB0liq1fsuszr?authuser=1 con rkf.valentini@gmail.com)
