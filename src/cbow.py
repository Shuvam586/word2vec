import numpy as np

def softmax(scores: np.ndarray) -> np.ndarray:
    new_scores = np.exp(scores-np.max(scores))  # stabilizing large scores 
    return new_scores/np.sum(new_scores)

class CBOW:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = np.random.randn(
            vocab_size,
            embedding_dim
        ) * 0.01

        self.W_out = np.random.randn(
            vocab_size,
            embedding_dim
        ) * 0.01
    
    def forward(self, context_ids):
        context_vectors = self.W_in[context_ids]
        h = np.mean(context_vectors, axis=0)

        scores = self.W_out @ h
        probs = softmax(scores)

        return h, probs
    
    def loss(self, probs, target_id):
        return -np.log(probs[target_id]+1e-12) # 1e-12 prevents log(0) = -inf


    def backward(self, context_ids, target_id, h, probs, learning_rate):
        dscores = probs.copy()
        dscores[target_id] -= 1

        # these 2 lines are of same rank??
        dW_out = np.outer(dscores, h)
        dh = self.W_out.T @ dscores

        dcontext = dh / len(context_ids)

        # updating W_out
        self.W_out -= learning_rate*dW_out

        # updating W_in
        np.add.at(
            self.W_in,
            context_ids,
            -learning_rate * dcontext
        )