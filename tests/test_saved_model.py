import src.model_io as model_io

model_name = input()

model, word_to_id, id_to_word = model_io.load_model(model_name)