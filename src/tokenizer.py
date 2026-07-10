from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tiny.txt"

file = "data/tiny"

with open(file+".txt", "r") as f:
    words = f.read().split()

words_set = set(words)

num = 0
min_count = 3
words_dict = {}
ids_dict = {}
counts_dict = {}

for w in list(words_set):
    wc = words.count(w)
    if wc >= min_count:
        words_dict[w] = num
        ids_dict[num] = w
        counts_dict[w] = wc
        num += 1

def word_to_id(word: str):
    try:
        return words_dict[word]
    except:
        return -1
    
def id_to_word(id: int):
    try:
        return ids_dict[id]
    except:
        return "-na-"

def counts(word: str):
    try:
        return counts_dict[word]
    except:
        return 0

def corpus_ids(sentence: list):
    ids = []
    for s in sentence.split():
        ids.append(word_to_id(s))
    return ids
