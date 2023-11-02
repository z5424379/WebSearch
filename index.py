import os
import sys
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import defaultdict

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('P'):
        return wordnet.ADJ_SAT
    else:
        return None

def process_document(document):
    document = re.sub('[^a-zA-Z0-9.,?!]', ' ', document)
    document = re.sub('(\d+)?\.\d+', '', document)
    words = word_tokenize(document) # 分词
    tags = nltk.pos_tag(words)
    wnl = WordNetLemmatizer() # 词形还原
    lemmas_words = []
    for tag in tags:
        wordnet_pos = get_pos(tag[1]) or wordnet.NOUN
        lemmas_words.append((wnl.lemmatize(tag[0], pos=wordnet_pos), wordnet_pos))
    words = [(word[0].lower().replace('.','').replace(',',''), word[1]) for word in lemmas_words]
    porter = PorterStemmer()
    words = [(porter.stem(word[0]), word[1]) for word in words if not re.match('^[a-z]$', word[0])] # 词干提取
    return words

def get_inverted_index(doc_folder, index_folder):
    index = defaultdict(list)
    doc_cnt, token_cnt, term_cnt = 0, 0, 0

    for filename in sorted(os.listdir(doc_folder)):
        doc_cnt += 1
        doc_id = filename
        with open(os.path.join(doc_folder, filename), 'r') as f:
            document = f.read()
            words = process_document(document)
            token_cnt += len(words)
            for position, word in enumerate(words):
                index[word].append(f"{doc_id}:{position}")
    
    with open(os.path.join(index_folder, 'index.txt'), 'w') as f:
        for word in index:
            f.write(f"{word[0]},{word[1]}")
            for position in index[word]:
                f.write(f" {position}")
            f.write("\n")
    
    term_cnt = len(index)
    print(f"Total number of documents: {doc_cnt}")
    print(f"Total number of tokens: {token_cnt}")
    print(f"Total number of terms: {term_cnt}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python index.py <folder-of-documents> <folder-of-indexes>")
        exit(1)

    doc_folder = sys.argv[1]
    index_folder = sys.argv[2]

    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    get_inverted_index(doc_folder, index_folder)



