from train_skipgram_hs import train_model
import src.analogy as analogy

model, word_to_id, id_to_word = train_model("tiny", 30, 0.1)

while True:
    worda = input("a: ")
    wordb = input("b: ")
    wordc = input("c: ")
    try:
        analogy.analogy(worda, wordb, wordc, word_to_id, id_to_word, model.W_in, 10)
    except:
        print(f"something went wrong with these words")