import random

def generate_cbow_examples(corpus_ids, window_size, random_window = False):
    examples = []

    C = window_size
    R = random.randint(1, C)
    
    if not random_window:
        R = C
    
    for index, target in enumerate(corpus_ids):
        left = max(0, index-R)
        right = min(len(corpus_ids), index+R+1)

        left_ctx = corpus_ids[left: index]
        right_ctx = corpus_ids[index+1: right]

        ctx = left_ctx+right_ctx

        examples.append([ctx, target])
    
    return examples

def generate_skipgram_examples(corpus_ids, window_size, random_window = False):
    examples = []

    C = window_size
    R = random.randint(1, C)

    if not random_window:
        R = C

    for index, target in enumerate(corpus_ids):
        left = max(0, index-R)
        right = min(len(corpus_ids), index+R+1)

        left_ctx = corpus_ids[left: index]
        right_ctx = corpus_ids[index+1: right]

        ctx = left_ctx+right_ctx

        for c in ctx:
            examples.append([target, c])

    return examples
