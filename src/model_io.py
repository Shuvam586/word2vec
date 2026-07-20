import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def save_model(name, model, word_to_id, id_to_word):
    data = {
        "model": model,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
    }

    filepath = PROJECT_ROOT / "models" / (name+".pkl")

    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_model(name):
    filepath = PROJECT_ROOT / "models" / (name+".pkl")
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return (
        data["model"],
        data["word_to_id"],
        data["id_to_word"],
    )