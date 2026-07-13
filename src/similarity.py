import numpy as np

def normalize_embeddings(W_in: np.ndarray) -> np.ndarray:
    new_W_in = W_in / np.linalg.norm(W_in, axis=1, keepdims=True)
    return new_W_in

def cosine_similarity(Va: np.ndarray, Vb: np.ndarray) -> float:
    return np.dot(Va, Vb)

def most_similar(word: str, word_to_id: dict, id_to_word: dict, W_in: np.ndarray, k: int = 10) -> list:
    word_id = word_to_id[word]
    W_in_norm = normalize_embeddings(W_in)

    query = W_in_norm[word_id]
    similarities = W_in_norm @ query

    similarities[word_id] = -np.inf

    standings = np.flip(np.argsort(similarities))
    
    similar_words = []

    for i in range(k):
        word = id_to_word[standings[i]]
        print(word, similarities[standings[i]])
        similar_words.append(word)
    
    return similar_words