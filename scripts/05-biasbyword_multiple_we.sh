
TARGETA="HE"
TARGETB="SHE"

# convert glove matrices from .bin to .npy
for i in $(seq 4 8); do
  VOCABFILE="embeddings/vocab-C$i-V20.txt"
  EMBEDFILE="embeddings/vectors-C$i-V20-W8-D1-D100-R0.05-E150-S1.bin"
  OUTFILE="embeddings/glove-C$i-V20-W8-D1-D100-R0.05-E150-S1.npy"
  python3 -u scripts/02-build_glove_matrix.py \
    -v $VOCABFILE -e $EMBEDFILE -o $OUTFILE
done

# compute cosine bias for glove and w2v
for i in $(seq 4 8); do
  VOCABFILE="embeddings/vocab-C$i-V20.txt"
  # w2v
  FILE_W2V="embeddings/w2v-C$i-V20-W8-D100-SG1.npy"
  python3 -u scripts/04-biasbyword_we.py \
    --vocab $VOCABFILE --matrix $FILE_W2V --a $TARGETA --b $TARGETB
  # glove
  FILE_GLOVE="embeddings/glove-C$i-V20-W8-D1-D100-R0.05-E150-S1.npy"
  python3 -u scripts/04-biasbyword_we.py \
    --vocab $VOCABFILE --matrix $FILE_GLOVE --a $TARGETA --b $TARGETB
done
