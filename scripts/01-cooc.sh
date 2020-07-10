#!/bin/bash

BASE_DIR="$(dirname $0)/.."
BUILD_DIR=$BASE_DIR/GloVe/build  # GloVe binaries are here

# CMD ARGS
CONFIG_FILE=$1
CORPORA_DIR=${2:-"$BASE_DIR/corpora"}
RESULTS_DIR=${3:-"$BASE_DIR/embeddings"}
MEMORY=${4:-"6"}
NUM_THREADS=${5:-"6"}
VERBOSE=${6:-"2"}

CORPORA=(
$CORPORA_DIR/simplewikiselect.txt
$CORPORA_DIR/test.txt
# $CORPORA_DIR/nytselect.txt
)

# DEFAULT
CORPUS_ID=0
VOCAB_MIN_COUNT=20
WINDOW_SIZE=8
VECTOR_SIZE=50
ETA=0.05
MAX_ITER=100
SEED=1
DISTANCE_WEIGHTING=0

# Overide default
if [[ -f $CONFIG_FILE ]]; then
  echo Loading config from file.
  source $CONFIG_FILE
fi

echo Setting up embedding:
echo CORPUS_ID = $CORPUS_ID
echo VOCAB_MIN_COUNT = $VOCAB_MIN_COUNT
echo WINDOW_SIZE = $WINDOW_SIZE
echo VECTOR_SIZE = $VECTOR_SIZE
echo ETA = $ETA
echo MAX_ITER = $MAX_ITER
echo SEED = $SEED
echo DISTANCE_WEIGHTING = $DISTANCE_WEIGHTING
echo

# Concat parameters
CORPUS=${CORPORA[$CORPUS_ID]}
VOCAB_PARAMS=C$CORPUS_ID-V$VOCAB_MIN_COUNT
COOC_PARAMS=$VOCAB_PARAMS-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING
EMBED_PARAMS=$COOC_PARAMS-D$VECTOR_SIZE-R$ETA-E$MAX_ITER-S$SEED

# Files
VOCAB_FILE=$RESULTS_DIR/vocab-$VOCAB_PARAMS.txt
OVERFLOW_FILE=$RESULTS_DIR/overflow-$COOC_PARAMS
COOC_FILE=$RESULTS_DIR/cooc-$COOC_PARAMS.bin
TEMP_FILE=$RESULTS_DIR/temp-$EMBED_PARAMS
SHUF_FILE=$RESULTS_DIR/shuf-$EMBED_PARAMS.bin
SAVE_FILE=$RESULTS_DIR/vectors-$EMBED_PARAMS

if [[ ! -f $VOCAB_FILE ]]; then
  echo "Building $VOCAB_FILE"
  $BUILD_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -max-vocab $MAX_VOCAB -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
else
  echo "Vocab file: $VOCAB_FILE exists. Skipping."
fi

if [[ ! -f $COOC_FILE ]]; then
  echo "Building $COOC_FILE"
  $BUILD_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -overflow-file $OVERFLOW_FILE -distance-weighting $DISTANCE_WEIGHTING < $CORPUS > $COOC_FILE
else
  echo "Cooc file: $COOC_FILE exists. Skipping."
fi
