from train_skipgram_hs import train_model
import src.similarity as similarity

model, word_to_id, id_to_word = train_model("tiny", 30, 0.1)

while True:
    word = input("word: ")
    try:
        similarity.most_similar(word, word_to_id, id_to_word, model.W_in, 10)
    except:
        print(f"something went wrong with '{word}'")