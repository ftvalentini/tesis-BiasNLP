import sys
import json
from os import path
from gensim.corpora.wikicorpus import WikiCorpus

wiki_path = sys.argv[1]
outname = sys.argv[2]
### para la nube:
# wiki_path = '/home/Fran/bucket/corpora/enwiki-20200701-pages-articles-multistream.xml.bz2'
# outname = "E:\\tesis-BiasNLP\\corpora\\enwikiselect"
###

# base_dir = ""
# base_dir = path.join(path.dirname(path.realpath(__file__)), path.pardir)
# wiki_filename = 'simplewiki-20200620-pages-articles-multistream.xml.bz2'
# wiki_path = path.join(base_dir, 'corpora', wiki_filename)
# outname = path.join(base_dir, 'corpora', 'simplewikiselect')

MAX_WC = 500_000_000
ARTICLE_MIN_WC = 200
ARTICLE_MAX_WC = 10000

index = []  # Save information about articles as they've been processed.
wiki = WikiCorpus(wiki_path, dictionary=True)  # dict=True avoids making vocab
wiki.metadata = True  # Want article titles

if __name__ == '__main__':
    ac = 0
    wc = 0
    selected = []
    line = 0
    with open(outname + ".txt", "w", encoding="utf-8") as f:
        for document in wiki.get_texts():
        # for i in range(num_articles):
            article, (id, name) = document
            art_len = len(article)
            if art_len >= ARTICLE_MIN_WC and art_len <= ARTICLE_MAX_WC:
                text = " ".join(article)
                wc += art_len
                ac += 1
                pos = f.tell()
                index.append({"id": id, "name": name, "wc": art_len, "line": line,
                              "byte": pos})
                f.write(bytes(text, 'utf-8').decode('utf-8') + '\n')
                line += 1
                print(f"document {ac}")
            if wc >= MAX_WC:
                break

    print("Selected", ac, "documents. (", wc, "tokens )")

    metadata = {
        "source": path.basename(wiki_path),
        "document_min_wc": ARTICLE_MIN_WC,
        "document_max_wc": ARTICLE_MAX_WC,
        "num_documents": ac,
        "num_words": wc,
        "fields": list(index[0].keys()),
        "index": index}

    with open(outname + ".meta.json", "w") as f:
        json.dump(metadata, f, indent=4)

    with open(outname + ".meta.txt", "w") as f:
        del metadata["index"]
        for key, val in metadata.items():
            f.write(key)
            f.write(": ")
            f.write(str(val))
            f.write("\n")
