# fixed param
OUTDIR=""
SG=1
SIZE=100
WINDOW=8
MINCOUNT=20
SEED=1

# get ID of each corpus
source scripts/corpora_dict

# iterate over corpora ids
for i in $(seq 4 8); do
  VOCABFILE="embeddings/vocab-C$i-V20.txt"
  CORPUSFILE=${CORPORA[$i]}
  # train w2v
  python3 -u scripts/02-train_word2vec.py \
    --id $i --corpus $CORPUSFILE --vocab $VOCABFILE --outdir $OUTDIR \
    --size $SIZE --window $WINDOW --count $MINCOUNT --sg $SG --seed $SEED
done
