
# words lists
LIST_A="HE"
LIST_B="SHE"

echo Saving GloVe matrices
for i in $(seq 4 8); do
  VOCABFILE="embeddings/vocab-C$i-V20.txt"
  EMBEDFILE="embeddings/vectors-C$i-V20-W8-D1-D100-R0.05-E150-S1.bin"
  MATFILE="embeddings/glove-C$i-V20-W8-D1-D100-R0.05-E150-S1.npy"
  python3 scripts/02-build_embeddings_matrix.py -v $VOCABFILE -e $EMBEDFILE -o $MATFILE
done

echo Running tests with GloVe
for i in $(seq 3 8); do
  VOCABFILE="embeddings/vocab-C$i-V20.txt"
  MATFILE="embeddings/glove-C$i-V20-W8-D1-D100-R0.05-E150-S1.npy"
  # csv with bias by word
  python3 -u test_distbyword.py $VOCABFILE $MATFILE $LIST_A $LIST_B
done

echo Running tests with w2v
for i in $(seq 3 8); do
  VOCABFILE="embeddings/vocab-C$i-V20.txt"
  MATFILE="embeddings/w2v-C$i-V20-W8-D100-SG0.npy"
  # csv with bias by word
  python3 -u test_distbyword.py $VOCABFILE $MATFILE $LIST_A $LIST_B
done



# # plots with stopwords (REVISAR QUE ANDA SI SE USA DMPI CON CORPUS 3 y WE CON OTRO)
# FILE_WE="results/csv/distbyword_glove-C${i}_MALE_SHORT-FEMALE_SHORT.csv"
# FILE_DPMI="results/csv/dpmibyword_C3_MALE_SHORT-FEMALE_SHORT.csv"
# python -u test_stopwords.py $FILE_DPMI $FILE_WE
# # plots of frequencies
# python -u test_frequency.py $VOCABFILE $MATFILE
