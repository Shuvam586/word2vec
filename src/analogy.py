import numpy as np
import src.similarity as similarity

def analogy(a, b, c, word_to_id, id_to_word, embeddings, k=10):
    
    norm_embeddings = similarity.normalize_embeddings(embeddings)
    
    vec_a = norm_embeddings[word_to_id[a]]
    vec_b = norm_embeddings[word_to_id[b]]
    vec_c = norm_embeddings[word_to_id[c]]

    x = vec_c - vec_a + vec_b
    x /= np.linalg.norm(x)

    probable_vectors = norm_embeddings @ x
    probable_vectors[word_to_id[a]] = -np.inf
    probable_vectors[word_to_id[b]] = -np.inf
    probable_vectors[word_to_id[c]] = -np.inf

    standings = np.flip(np.argsort(probable_vectors))

    similar_words = []

    for i in range(k):
        word = id_to_word[standings[i]]
        print(word, probable_vectors[standings[i]])
        similar_words.append(word)

    return similar_words