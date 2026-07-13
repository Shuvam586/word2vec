import numpy as np
import src.cbow as cbow
import src.vocab as vocab
import src.analogy as analogy
import src.dataset as dataset
import src.tokenizer as tokenizer
import src.similarity as similarity
import src.huffman as huffman

corpus_size = input("[tiny, small, medium, raw]: ")
corpus_presets = ['medium', 'raw', 'small', 'tiny']

if corpus_size not in corpus_presets:
    print(f'{corpus_size} doesnt exist lmao')
else:
    tokens = tokenizer.load_and_tokenize(corpus_size, lowercase=True)
    
    word_to_id, id_to_word, word_counts, corpus_ids = vocab.build_vocab(tokens)

    # root = huffman.build_huffman_tree(word_counts)
    # huffman.generate_codes(root)

    examples = dataset.generate_cbow_examples(corpus_ids, window_size=2)

    model = cbow.CBOW(
        vocab_size=len(word_to_id),
        embedding_dim=50
    )

    epochs = int(input("epochs: "))
    learning_rate = float(input("lr: "))

    for epoch in range(epochs):
        
        # forwar -> loss -> backward

        total_loss = 0
        example_index = 1

        modified_lr = learning_rate

        for context, target in examples:

            progress = (epoch*len(examples)+example_index)/(epochs*len(examples))

            modified_lr = learning_rate * (1 - progress)

            h, probs = model.forward(context)

            loss = model.loss(probs, target)

            model.backward(
                context,
                target,
                h,
                probs,
                modified_lr
            )

            total_loss += loss
        
        print(f"epoch {epoch+1}/{epochs} | loss: {total_loss}")

    while True:

        try:

            word = input()
            print(f"most similar to {word}:")
            similarity.most_similar(word, word_to_id, id_to_word, model.W_in, 10)

            worda = input("a: ")
            wordb = input("b: ")
            wordc = input("c: ")

            analogy.analogy(worda, wordb, wordc, word_to_id, id_to_word, model.W_in)
        except KeyError:
            print('\nthat must have caused an error\n')