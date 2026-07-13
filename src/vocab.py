def build_vocab(tokens, min_count = 2):
    legit_words = []
    
    for i in list(set(tokens)):
        if tokens.count(i)>=min_count:
            legit_words.append(i)
    
    word_to_id = {}
    id_to_word = {}
    word_counts = {}

    for id, word in enumerate(legit_words):
        word_to_id[word] = id
        id_to_word[id] = word

        word_counts[word] = tokens.count(word)
        
    corpus_ids = []

    for t in tokens:
        if t in word_counts:
            corpus_ids.append(word_to_id[t])


    # this is temp, just to get the most frequently used words
    temp = [[word_counts[i], i] for i in word_counts]
    temp.sort(reverse=True)
    print(temp[:20])

    return word_to_id, id_to_word, word_counts, corpus_ids