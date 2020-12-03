
CORPUSFILE="corpora/enwikiselect.txt"
VOCABFILE="embeddings/vocab-C3-V20.txt"
SEEDS=(93 94 95 96 97)

for i in ${SEEDS[@]}; do
  SEED=$i
  OUTFILE="corpora/enwikiselect_newwords_S$SEED.txt"
  nohup python3 -u scripts/insert_new_words.py \
    $CORPUSFILE $VOCABFILE $OUTFILE $SEED
done
