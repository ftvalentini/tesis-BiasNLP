
# iterate over corpora ids
for i in $(seq 4 8); do
  line="CORPUS_ID=$i"
  # replace first line of config file
  sed -i "1s/.*/$line/" scripts/glove.config
  # train glove
  scripts/01-embed.sh scripts/glove.config
done
