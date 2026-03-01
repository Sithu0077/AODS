import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score

# ==========================================================
# COMPARISON: Batch GD vs Stochastic GD vs Mini-Batch GD
# ==========================================================

# ----------------------------------------------------------
# STEP 1: Load and Scale Dataset
# ----------------------------------------------------------
# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Scale features (important for gradient-based methods)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# STEP 2: Batch Gradient Descent (Closed-Form Solution)
# ----------------------------------------------------------
# LinearRegression uses Normal Equation (not iterative GD)
# It computes optimal weights directly.

batch_model = LinearRegression()
batch_model.fit(X_scaled, y)

# Predictions
batch_pred = batch_model.predict(X_scaled)
batch_r2 = r2_score(y, batch_pred)

# ----------------------------------------------------------
# STEP 3: Stochastic Gradient Descent (SGD)
# ----------------------------------------------------------
# Updates parameters using ONE sample at a time

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_scaled, y)

sgd_pred = sgd_model.predict(X_scaled)
sgd_r2 = r2_score(y, sgd_pred)

# ----------------------------------------------------------
# STEP 4: Mini-Batch Gradient Descent (Manual Simulation)
# ----------------------------------------------------------
# Updates parameters using small batches of data

mini_batch_model = SGDRegressor(max_iter=1, tol=None, random_state=42)

batch_size = 32
n_samples = X_scaled.shape[0]
epochs = 50

for epoch in range(epochs):

    # Shuffle dataset each epoch
    indices = np.random.permutation(n_samples)
    X_shuffled = X_scaled[indices]
    y_shuffled = y[indices]

    # Process data in mini-batches
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # partial_fit updates weights incrementally
        mini_batch_model.partial_fit(X_batch, y_batch)

# Evaluate mini-batch model
mini_pred = mini_batch_model.predict(X_scaled)
mini_r2 = r2_score(y, mini_pred)

# ----------------------------------------------------------
# STEP 5: Display Results Clearly
# ----------------------------------------------------------

print("Comparison of Gradient Descent Methods\n")

print("Batch Gradient Descent (Closed-Form)")
print("  Method        : Uses full dataset at once")
print(f"  R2 Score      : {batch_r2:.4f}")
print("  Convergence   : Stable but computationally expensive\n")

print("Stochastic Gradient Descent (SGD)")
print("  Method        : One sample per update")
print(f"  Iterations    : {sgd_model.n_iter_}")
print(f"  R2 Score      : {sgd_r2:.4f}")
print("  Convergence   : Fast but noisy\n")

print("Mini-Batch Gradient Descent")
print(f"  Batch Size    : {batch_size}")
print(f"  Epochs        : {epochs}")
print(f"  R2 Score      : {mini_r2:.4f}")
print("  Convergence   : Balanced (stable + efficient)\n")