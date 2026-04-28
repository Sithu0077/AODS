import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for _ in range(epochs):
    y_pred = m * X + b
    error = y_pred - y

    # Gradients
    dm = (2/n) * np.sum(X * error)
    db = (2/n) * np.sum(error)

    # Update
    m -= learning_rate * dm
    b -= learning_rate * db

print("Slope:", m)
print("Intercept:", b)

# Plot
plt.scatter(X, y, color='red')
plt.plot(X, m*X + b, color='blue')
plt.show()