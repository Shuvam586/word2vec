def generate_cbow_examples(corpus_ids, window_size):
    examples = []
    
    for index, target in enumerate(corpus_ids):
        left = max(0, index-window_size)
        right = min(len(corpus_ids), index+window_size+1)

        left_ctx = corpus_ids[left: index]
        right_ctx = corpus_ids[index+1: right]

        ctx = left_ctx+right_ctx

        examples.append([ctx, target])
    
    return examples

def generate_skipgram_examples(corpus_ids, window_size):
    examples = []

    for index, target in enumerate(corpus_ids):
        left = max(0, index-window_size)
        right = min(len(corpus_ids), index+window_size+1)

        left_ctx = corpus_ids[left: index]
        right_ctx = corpus_ids[index+1: right]

        ctx = left_ctx+right_ctx

        for c in ctx:
            examples.append([target, c])

    return examples
