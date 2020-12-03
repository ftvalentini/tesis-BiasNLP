
IDS=(93 94 95 96 97)

# iterate over corpora ids
for i in ${IDS[@]}; do
  line="CORPUS_ID=$i"
  # replace first line of config file
  sed -i "1s/.*/$line/" scripts/glove.config
  # train glove
  scripts/01-glove.sh scripts/glove.config
done
