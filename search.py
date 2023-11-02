import sys
import os
import nltk
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
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

def process_words(words):
    tags = nltk.pos_tag(words)
    wnl = WordNetLemmatizer() # 词形还原
    lemmas_words = []
    for tag in tags:
        wordnet_pos = get_pos(tag[1]) or wordnet.NOUN
        lemmas_words.append((wnl.lemmatize(tag[0], pos=wordnet_pos), wordnet_pos))
    words = [(word[0].lower().replace('.','').replace(',',''), word[1]) for word in lemmas_words]
    porter = PorterStemmer()
    words = [(porter.stem(word[0]), word[1]) for word in words if not re.match('^[a-z]$', word[0])]
    return words

def get_invert_index(index_folder):
    invert_index = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(index_folder, 'index.txt'), 'r') as f:
        for line in f:
            words = line.split()
            word, pos = words[0].split(',')
            word = (word, pos)
            for doc_id in words[1:]:
                doc_id, position = map(int, doc_id.split(':'))
                invert_index[word][doc_id].append(position)
            # 对于每个词，按照文档编号从小到大排序
            invert_index[word] = dict(sorted(invert_index[word].items(), key=lambda x: x[0]))
    return invert_index

# 找公共的doc_id
def find_common_doc_id(words):
    ans = set(invert_index[words[0]].keys())
    for word in words[1:]:
        print(ans)
        ans &= set(invert_index[word].keys())
    return ans

def find_min_distance(word1, word2, doc_id):
    min_distance = float('inf')
    correct_order = True
    for position1 in invert_index[word1][doc_id]:
        for position2 in invert_index[word2][doc_id]:
            if abs(position1 - position2)-1 < min_distance:
                min_distance = abs(position1 - position2)-1
                correct_order = position1 < position2
    return min_distance, correct_order

def distances(doc_id, words):
    total_distance = 0
    total_correct = 0
    for i in range(len(words)-1):
        min_distance, correct_order = find_min_distance(words[i], words[i+1], doc_id)
        total_distance += min_distance
        if correct_order:
            total_correct += 1
    return total_distance, total_correct

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python search.py <folder-of-indexes>")
        exit(1)

    index_folder = sys.argv[1]
    invert_index = get_invert_index(index_folder)
    
    for query in sys.stdin:
        words = query.split()
        words = process_words(words)
        print(words)
        common_doc = find_common_doc_id(words)
        if not common_doc:
            print("Not found")
            continue
        doc_distances = {doc_id: distances(doc_id, words) for doc_id in common_doc}
        print(doc_distances)
        common_doc = sorted(common_doc, key=lambda x: (doc_distances[x][0], -doc_distances[x][1], x))
        for doc_id in common_doc:
            print(doc_id)