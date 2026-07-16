import numpy as np
import src.hierarchical_softmax as hs

h = np.random.randn(50)

word_to_code = {
    0: [0,1],
    1: [1,0]
}

word_to_path = {
    0: [0,2],
    1: [1,2]
}

hs = hs.HierarchicalSoftmax(
    embedding_dim=50,
    num_internal_nodes=3, 
    word_to_code=word_to_code, 
    word_to_path=word_to_path
)

for i in range(100):
    scores, prob = hs.forward(h=h, target_id=0)

    loss = hs.loss(probabilities=prob, target_id=0)

    hs.backward(
        h=h,
        target_id=0,
        probabilities=prob,
        learning_rate=0.1
    )

    print(f"{i+1}/100: {loss}")