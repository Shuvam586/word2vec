import argparse
import numpy as np
import src.vocab as vocab
import src.analogy as analogy
import src.cbow_hs as cbow_hs
import src.huffman as huffman
import src.dataset as dataset
import src.model_io as model_io
import src.tokenizer as tokenizer
import src.similarity as similarity

parser = argparse.ArgumentParser(
    description="cbow hs"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=30
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.1
)

parser.add_argument(
    "--size",
    type=str,
    default='tiny.txt'
)

args = parser.parse_args()

def train_model(corpus_size, epochs, learning_rate):
    corpus_presets = ['medium', 'raw', 'small', 'tiny']
    if corpus_size not in corpus_presets:
        return f'{corpus_size} doesnt exist lmao'
    else:
        print(f"cbow_hs:\nepochs: {epochs}  |  lr: {learning_rate}  |  corpus_size: {corpus_size}")
        
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

            examples_len = len(examples)
            examples_by10 = int(len(examples)/3)

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

                if (example_index%examples_by10==0):
                    print(f"example {((example_index / examples_len)*100):.2f}%")
            
            print(f"epoch {epoch+1}/{epochs} | loss: {total_loss}")
        
        model_name = f"cbow_hs_{corpus_size}_{epochs}_{str(learning_rate).replace(".", "dot")}"
        
        model_io.save_model(
            name=model_name,
            model=model,
            word_to_id=word_to_id,
            id_to_word=id_to_word
        )

        return model, word_to_id, id_to_word


# corpus_size = input("[tiny, small, medium, raw]: ")

train_model(
    corpus_size=args.size,
    epochs=args.epochs,
    learning_rate=args.lr
)