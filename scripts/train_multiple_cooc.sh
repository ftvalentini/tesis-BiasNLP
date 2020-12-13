
IDS=(4 5 6 7 8)

# get co-occurrence params (VOCAB_MIN_COUNT,WINDOW_SIZE,DISTANCE_WEIGHTING)
source scripts/cooc.config

chmod +x scripts/01-cooc.sh

# iterate over corpora ids
for i in ${IDS[@]}; do

  # replace first line of config file
  line="CORPUS_ID=$i"
  sed -i "1s/.*/$line/" scripts/cooc.config

  # build cooc matrix
  scripts/01-cooc.sh scripts/cooc.config

  # from cooc. bin to sparse cooc. and PMI matrices
  VOCABFILE="embeddings/vocab-C$i-V$VOCAB_MIN_COUNT.txt"
  BINFILE="embeddings/cooc-C$i-V$VOCAB_MIN_COUNT-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING.bin"
  COOCFILE="embeddings/cooc-C$i-V$VOCAB_MIN_COUNT-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING.npz"
  PMIFILE="embeddings/pmi-C$i-V$VOCAB_MIN_COUNT-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING.npz"
  python3 -u scripts/02-build_cooc_matrix.py -v $VOCABFILE -c $BINFILE -o $COOCFILE
  python3 -u scripts/03-build_pmi_matrix.py --matrix $COOCFILE --outfile $PMIFILE

done
