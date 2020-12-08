
IDS=(4 5 6 7 8)

# read GloVe params
# (VOCAB_MIN_COUNT,WINDOW_SIZE,VECTOR_SIZE,ETA,MAX_ITER,SEED,DISTANCE_WEIGHTING)
source scripts/glove.config

# iterate over corpora ids
for i in ${IDS[@]}; do

  # replace first line of config file
  line="CORPUS_ID=$i"
  sed -i "1s/.*/$line/" scripts/glove.config

  # train glove
  scripts/01-glove.sh scripts/glove.config

  # extract GloVe matrices into numpy format
  VOCABFILE="embeddings/vocab-C$i-V$VOCAB_MIN_COUNT.txt"
  BINFILE="embeddings/vectors-C$i-V$VOCAB_MIN_COUNT-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING-D$VECTOR_SIZE-R$ETA-E$MAX_ITER-S$SEED.bin"
  MATFILE="embeddings/full_vectors-C$i-V$VOCAB_MIN_COUNT-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING-D$VECTOR_SIZE-R$ETA-E$MAX_ITER-S$SEED.pkl"
  EMBEDFILE="embeddings/glove-C$i-V$VOCAB_MIN_COUNT-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING-D$VECTOR_SIZE-R$ETA-E$MAX_ITER-S$SEED.npy"
  python3 -u scripts/02-build_glove_matrices.py $VOCABFILE $BINFILE $MATFILE $EMBEDFILE

done
