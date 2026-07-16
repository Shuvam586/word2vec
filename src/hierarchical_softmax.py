import numpy as np
from src.utils import sigmoid

class HierarchicalSoftmax:
    def __init__(self, embedding_dim, num_internal_nodes, word_to_code, word_to_path):
        self.word_to_code = word_to_code
        self.word_to_path = word_to_path
        self.W_tree = np.random.randn(num_internal_nodes, embedding_dim) * 0.01
    
    def forward(self, h, target_id):
        path = self.word_to_path[target_id]
        code = self.word_to_code[target_id]

        scores = []
        probabilities = []

        for node in path:
            # print("node in path: ", path)
            node_vector = self.W_tree[node]

            score = np.dot(h, node_vector)
            p = sigmoid(score)

            scores.append(score)
            probabilities.append(p)

        return scores, probabilities
    
    def loss(self, probabilities, target_id):
        loss = 0

        code = self.word_to_code[target_id]

        for p, bit in zip(probabilities, code):
            if bit==1:
                loss -= np.log(p)
            else:
                loss -= np.log(1-p)
        
        return loss
    
    def backward(self, h, target_id, probabilities, learning_rate):
        path = self.word_to_path[target_id]
        code = self.word_to_code[target_id]

        dh = np.zeros_like(h)

        for node, bit, p in zip(path, code, probabilities):

            delta = p - bit

            dW = delta * h

            dh += delta * self.W_tree[node]

            self.W_tree[node] -= learning_rate * dW

        return dh