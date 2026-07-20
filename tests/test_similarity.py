import src.model_io as model_io
import src.similarity as similarity

model, word_to_id, id_to_word = model_io.load_model(input("model name: "))

while True:
    word = input("word: ")
    try:
        similarity.most_similar(word, word_to_id, id_to_word, model.W_in, 10)
    except:
        print(f"something went wrong with '{word}'")