import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==================================================
# OVERFITTING vs UNDERFITTING DEMONSTRATION
# ==================================================

# --------------------------------------------------
# STEP 1: Generate Synthetic Non-Linear Dataset
# --------------------------------------------------
# X values from -3 to 3
X = np.linspace(-3, 3, 100).reshape(-1, 1)

# True function: sin(x)
# Add random noise to simulate real-world data
y = np.sin(X).ravel() + np.random.randn(100) * 0.2

# --------------------------------------------------
# STEP 2: Split Data into Train and Test Sets
# --------------------------------------------------
# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# --------------------------------------------------
# STEP 3: Train Models with Different Complexities
# --------------------------------------------------
# Degree 1  → Very simple model
# Degree 3  → Moderate complexity
# Degree 10 → Very complex model

degrees = [1, 3, 10]

print("Overfitting vs Underfitting Analysis\n")

for degree in degrees:

    # Create a pipeline:
    # 1) Generate polynomial features
    # 2) Apply Linear Regression
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate performance using R² score
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Display results
    print(f"Model Complexity: Polynomial Degree {degree}")
    print(f"  Training Score : {train_score:.3f}")
    print(f"  Testing Score  : {test_score:.3f}")

    # --------------------------------------------------
    # Interpretation Logic
    # --------------------------------------------------

    # Underfitting: Low training performance
    if train_score < 0.7:
        print("  Interpretation : Underfitting (model too simple)\n")

    # Overfitting: High train score but poor test score
    elif train_score > 0.95 and test_score < train_score:
        print("  Interpretation : Overfitting (model too complex)\n")

    # Balanced model
    else:
        print("  Interpretation : Good Fit (balanced model)\n")