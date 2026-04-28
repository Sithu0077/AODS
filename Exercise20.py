import numpy as np

# Activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Data (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)

# Weights
w1 = np.random.rand(2, 2)
w2 = np.random.rand(2, 1)

# Training
for epoch in range(10000):
    # Forward
    h = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(h, w2))

    # Error
    error = y - output

    # Backprop
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(w2.T) * sigmoid_derivative(h)

    # Update
    w2 += h.T.dot(d_output)
    w1 += X.T.dot(d_hidden)

print(output)