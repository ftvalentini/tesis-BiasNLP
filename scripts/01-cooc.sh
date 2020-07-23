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
$CORPORA_DIR/simplewikiselect.txt # ID 0
$CORPORA_DIR/test.txt # ID 1
$CORPORA_DIR/test_short.txt # ID 2
$CORPORA_DIR/enwikiselect.txt # ID 3
)

# DEFAULT
CORPUS_ID=0
VOCAB_MIN_COUNT=1
WINDOW_SIZE=8
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
echo DISTANCE_WEIGHTING = $DISTANCE_WEIGHTING
echo

# Concat parameters
CORPUS=${CORPORA[$CORPUS_ID]}
VOCAB_PARAMS=C$CORPUS_ID-V$VOCAB_MIN_COUNT
COOC_PARAMS=$VOCAB_PARAMS-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING

# Files
VOCAB_FILE=$RESULTS_DIR/vocab-$VOCAB_PARAMS.txt
OVERFLOW_FILE=$RESULTS_DIR/overflow-$COOC_PARAMS
COOC_FILE=$RESULTS_DIR/cooc-$COOC_PARAMS.bin

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
