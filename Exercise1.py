# ============================================================
# FEATURE SCALING vs NO SCALING
# Effect on SGD Regressor Convergence and Performance
# ============================================================

# ------------------------------------------------------------
# STEP 0: Import Required Libraries
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")  # Hide convergence warnings for clarity

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------
# X -> Input features (10 numerical medical features)
# y -> Target variable (disease progression measure)

X, y = load_diabetes(return_X_y=True)

print("Dataset Shape:")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("-" * 50)


# ------------------------------------------------------------
# STEP 2: Split Dataset into Train and Test Sets
# ------------------------------------------------------------
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)
print("-" * 50)


# ------------------------------------------------------------
# STEP 3: Train SGD Model WITHOUT Feature Scaling
# ------------------------------------------------------------
# If features have different ranges,
# gradient descent becomes inefficient

model_no_scale = SGDRegressor(
    max_iter=1000,    # Maximum number of iterations
    tol=1e-3,         # Stopping tolerance
    random_state=42
)

# Train model
model_no_scale.fit(X_train, y_train)

# Predict on test data
y_pred_no_scale = model_no_scale.predict(X_test)

# Evaluate model
r2_no_scale = r2_score(y_test, y_pred_no_scale)

print("WITHOUT Feature Scaling:")
print("Iterations used =", model_no_scale.n_iter_)
print("R2 Score =", round(r2_no_scale, 4))
print("-" * 50)


# ------------------------------------------------------------
# STEP 4: Apply Feature Scaling (Standardization)
# ------------------------------------------------------------
# StandardScaler converts:
# mean = 0
# standard deviation = 1

scaler = StandardScaler()

# Fit only on training data (IMPORTANT!)
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using same scaler
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# STEP 5: Train SGD Model WITH Feature Scaling
# ------------------------------------------------------------

model_scaled = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Train model
model_scaled.fit(X_train_scaled, y_train)

# Predict on scaled test data
y_pred_scaled = model_scaled.predict(X_test_scaled)

# Evaluate model
r2_scaled = r2_score(y_test, y_pred_scaled)

print("WITH Feature Scaling:")
print("Iterations used =", model_scaled.n_iter_)
print("R2 Score =", round(r2_scaled, 4))
print("-" * 50)


# ------------------------------------------------------------
# STEP 6: Final Comparison Summary
# ------------------------------------------------------------

print("FINAL COMPARISON")
print("=" * 50)

print("Without Scaling:")
print("  Iterations:", model_no_scale.n_iter_)
print("  R2 Score  :", round(r2_no_scale, 4))

print("\nWith Scaling:")
print("  Iterations:", model_scaled.n_iter_)
print("  R2 Score  :", round(r2_scaled, 4))

print("\nINTERPRETATION:")
print("Feature scaling improves convergence speed and")
print("usually improves model stability and performance.")
print("=" * 50)