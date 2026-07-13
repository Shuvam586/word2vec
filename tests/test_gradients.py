# run with:
# python3 -m tests.test_gradients

import src.cbow as cbow

model = cbow.CBOW(5, 3)
eps = 1e-3
original = model.W_out[2][1]


# plus epsilon
model.W_out[2][1] = original + eps
context = [1, 3]

h, probs = model.forward(context)

loss_plus = model.loss(probs, target_id=2)


# minus epsilon
model.W_out[2][1] = original - eps
context = [1, 3]

h, probs = model.forward(context)

loss_minus = model.loss(probs, target_id=2)


# restoring og val 
numerical_grad = (loss_plus-loss_minus)/(2*eps)

model.W_out[2][1] = original
h, probs = model.forward(context)

dW_out = model.backward(
    context,
    target_id=2,
    h=h,
    probs=probs,
    learning_rate=0.001
)

analytical_grad = dW_out[2][1]

print(f"Numerical Grad: {numerical_grad}\nAnalytical Grad: {analytical_grad}")
print(f"Diff: {numerical_grad-analytical_grad}")