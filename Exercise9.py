import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# ==========================================================
# MACHINE LEARNING PIPELINE WITH CROSS-VALIDATION
# ==========================================================

# ----------------------------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------------------------
# Using built-in diabetes dataset
# No internet dependency
X, y = load_diabetes(return_X_y=True)

# ----------------------------------------------------------
# STEP 2: Create Machine Learning Pipeline
# ----------------------------------------------------------
# Pipeline ensures:
# 1. Scaling is applied properly
# 2. No data leakage during cross-validation
# 3. Clean and reproducible workflow

pipeline = Pipeline([
    ("scaler", StandardScaler()),   # Step 1: Feature scaling
    ("model", Ridge(alpha=1.0))     # Step 2: Ridge regression (L2 regularization)
])

# ----------------------------------------------------------
# STEP 3: Perform Cross-Validation
# ----------------------------------------------------------
# 5-fold cross-validation
# scoring="r2" measures regression performance

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="r2"
)

# ----------------------------------------------------------
# STEP 4: Display Results Clearly
# ----------------------------------------------------------

print("Machine Learning Pipeline Optimization\n")

print("Cross-Validation Scores:")
for i, score in enumerate(cv_scores, start=1):
    print(f"  Fold {i} Score : {score:.3f}")

print(f"\nAverage CV Score : {cv_scores.mean():.3f}")

# ----------------------------------------------------------
# INTERPRETATION
# ----------------------------------------------------------
print("\nInterpretation:")
print("- Scaling stabilizes optimization for linear models.")
print("- Ridge regularization (L2) reduces overfitting.")
print("- Cross-validation provides robust generalization estimate.")