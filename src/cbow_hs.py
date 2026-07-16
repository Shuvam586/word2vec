import numpy as np
import src.hierarchical_softmax as hs

def softmax(scores: np.ndarray) -> np.ndarray:
    new_scores = np.exp(scores-np.max(scores))  # stabilizing large scores 
    return new_scores/np.sum(new_scores)

class CBOW_HS:
    def __init__(self, vocab_size, embedding_dim, word_to_code, word_to_path):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = np.random.randn(
            vocab_size,
            embedding_dim
        ) * 0.01

        self.hs = hs.HierarchicalSoftmax(
            embedding_dim=embedding_dim,
            num_internal_nodes=vocab_size,
            word_to_code=word_to_code,
            word_to_path=word_to_path
        )
    
    def forward(self, context_ids, target_id):
        context_vectors = self.W_in[context_ids]
        h = np.mean(context_vectors, axis=0)

        scores, probs = self.hs.forward(
            h,
            target_id
        )

        return h, probs
    
    def loss(self, probs, target_id):
        return self.hs.loss(
            probabilities=probs,
            target_id=target_id
        )

    def backward(self, context_ids, target_id, h, probs, learning_rate):
        dh = self.hs.backward(
            h=h,
            target_id=target_id,
            probabilities=probs,
            learning_rate=learning_rate
        )

        dcontext = dh / len(context_ids)

        np.add.at(
            self.W_in,
            context_ids,
            -learning_rate * dcontext
        )