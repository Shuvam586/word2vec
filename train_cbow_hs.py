import numpy as np
import src.vocab as vocab
import src.analogy as analogy
import src.cbow_hs as cbow_hs
import src.huffman as huffman
import src.dataset as dataset
import src.tokenizer as tokenizer
import src.similarity as similarity

def train_model(corpus_size, epochs = 30, learning_rate = 0.1):
    corpus_presets = ['medium', 'raw', 'small', 'tiny']
    if corpus_size not in corpus_presets:
        return f'{corpus_size} doesnt exist lmao'
    else:
        tokens = tokenizer.load_and_tokenize(corpus_size, lowercase=True)
        
        word_to_id, id_to_word, word_counts, corpus_ids = vocab.build_vocab(tokens)

        root = huffman.build_huffman_tree(word_counts)
        word_to_code, word_to_path = huffman.generate_codes(root, word_to_id), huffman.generate_paths(root, word_to_id)

        examples = dataset.generate_cbow_examples(corpus_ids, window_size=2)

        # model = cbow.CBOW(
        #     vocab_size=len(word_to_id),
        #     embedding_dim=50
        # )

        model = cbow_hs.CBOW_HS(
            vocab_size=len(word_to_id),
            embedding_dim=50,
            word_to_code=word_to_code,
            word_to_path=word_to_path
        )

        for epoch in range(epochs):
            
            # forwar -> loss -> backward

            total_loss = 0
            example_index = 1

            modified_lr = learning_rate

            for context, target in examples:

                progress = (epoch*len(examples)+example_index)/(epochs*len(examples))

                modified_lr = learning_rate * (1 - progress)

                h, probs = model.forward(context, target)

                loss = model.loss(probs, target)

                model.backward(
                    context,
                    target,
                    h,
                    probs,
                    modified_lr
                )

                example_index += 1
                total_loss += loss
            
            print(f"epoch {epoch+1}/{epochs} | loss: {total_loss}")
        
        return model, word_to_id, id_to_word


# corpus_size = input("[tiny, small, medium, raw]: ")

# train_model(corpus_size)