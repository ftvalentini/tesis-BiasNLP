
IDS=(4 5 6 7 8)

### Fixed param
# GloVe and word2vec
SIZE=100 # vector dimension
WINDOW=5 # window size
MINCOUNT=20 # vocab min count
SEED=1 # random seed
# GloVe
ETA=0.05 # initial learning rate
ITER=50 # max iterations
DIST_GLOVE=1 # distance weighting
# word2vec
SG=1 # skipgram
# cooc
DIST_COOC = 0
###

# target words
TARGETA=${1:-"HE"}
TARGETB=${2:-"SHE"}


# iterate over corpora IDs
for i in ${IDS[@]}; do

  VOCABFILE="embeddings/vocab-C$i-V$MINCOUNT.txt"

  # PMI-based biases
  COOCFILE="embeddings/cooc-C$i-V$MINCOUNT-W$WINDOW-D$DIST_COOC.npz"
  PMIFILE="embeddings/pmi-C$i-V$MINCOUNT-W$WINDOW-D$DIST_COOC.npz"
  python3 -u scripts/04-biasbyword_cooc.py \
    --vocab $VOCABFILE --cooc $COOCFILE --pmi $PMIFILE --a $TARGETA --b $TARGETB

  # w2v bias
  FILE_W2V="embeddings/w2v-C$i-V$MINCOUNT-W$WINDOW-D$SIZE-SG$SG-S$SEED.npy"
  python3 -u scripts/04-biasbyword_we.py \
  --vocab $VOCABFILE --matrix $FILE_W2V --a $TARGETA --b $TARGETB

  # glove
  FILE_GLOVE="embeddings/glove-C$i-V$MINCOUNT-W$WINDOW-D$DIST_GLOVE-D$SIZE-R$ETA-E$ITER-S$SEED.npy"
  python3 -u scripts/04-biasbyword_we.py \
    --vocab $VOCABFILE --matrix $FILE_GLOVE --a $TARGETA --b $TARGETB

  # join all
  python3 -u scripts/05-biasbyword_join.py --corpus $i --a $TARGETA --b $TARGETB

done
