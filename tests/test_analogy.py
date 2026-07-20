import src.analogy as analogy
import src.model_io as model_io

model, word_to_id, id_to_word = model_io.load_model(input("model name: "))

while True:
    worda = input("a: ")
    wordb = input("b: ")
    wordc = input("c: ")
    try:
        analogy.analogy(worda, wordb, wordc, word_to_id, id_to_word, model.W_in, 10)
    except:
        print(f"something went wrong with these words")