import numpy as np

# ==================================================
# LINEAR REGRESSION USING GRADIENT DESCENT
# ==================================================

# --------------------------------------------------
# STEP 1: Generate Synthetic Dataset
# --------------------------------------------------
# Create input values from 0 to 10 (100 points)
X = np.linspace(0, 10, 100)

# True relationship: y = 3x + 5 + noise
# np.random.randn adds Gaussian noise
y = 3 * X + 5 + np.random.randn(100)

# --------------------------------------------------
# STEP 2: Initialize Parameters
# --------------------------------------------------
# w -> weight (slope)
# b -> bias (intercept)
w = 0.0
b = 0.0

# Learning rate controls step size of updates
lr = 0.01

# Number of iterations (epochs)
epochs = 1000


# --------------------------------------------------
# STEP 3: Gradient Descent Optimization Loop
# --------------------------------------------------
for i in range(epochs):

    # Forward pass: predict output
    y_pred = w * X + b

    # --------------------------------------------------
    # Compute Gradients
    # --------------------------------------------------
    # Mean Squared Error loss:
    # L = (1/n) * Σ (y - y_pred)^2
    #
    # Partial derivative w.r.t w:
    # dw = -2 * mean(X * (y - y_pred))
    #
    # Partial derivative w.r.t b:
    # db = -2 * mean(y - y_pred)

    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    # --------------------------------------------------
    # Update Parameters
    # --------------------------------------------------
    # Move parameters in opposite direction of gradient
    w -= lr * dw
    b -= lr * db


# --------------------------------------------------
# STEP 4: Display Results
# --------------------------------------------------
print("Linear Regression Model after Training\n")

print("Learned Parameters:")
print(f"  Weight (w) : {w:.2f}")
print(f"  Bias (b)   : {b:.2f}\n")

print("Interpretation:")
print("  The model has learned a linear relationship of the form:")
print("      y = w*x + b")
print("  where:")
print("      w = slope of the line")
print("      b = y-intercept")