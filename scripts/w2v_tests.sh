
# iterate over corpora ids
for i in {4..8}; do
  # parameters
  VOCABFILE="embeddings/vocab-C$i-V20.txt"
  MATFILE="embeddings/w2v-C$i-V20-W8-D100-SG0.npy"
  FILE_DPMI="results/csv/dpmibyword_C3_MALE_SHORT-FEMALE_SHORT.csv"
  FILE_WE="results/csv/distbyword_w2v-C${i}_MALE_SHORT-FEMALE_SHORT.csv"
  # csv with bias by word
  python -u test_distbyword.py $VOCABFILE $MATFILE
  # plots with stopwords (REVISAR QUE ANDA SI SE USA DMPI CON CORPUS 3 y WE CON OTRO)
  python -u test_stopwords.py $FILE_DPMI $FILE_WE
  # plots of frequencies
  python -u test_frequency.py $VOCABFILE $MATFILE
done
