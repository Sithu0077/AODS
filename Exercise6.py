import warnings
warnings.filterwarnings("ignore")  # Suppress convergence warnings

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==================================================
# HINGE LOSS vs LOGISTIC LOSS COMPARISON
# ==================================================

# --------------------------------------------------
# STEP 1: Generate a Binary Classification Dataset
# --------------------------------------------------
# n_samples     : total number of samples
# n_features    : total number of input features
# n_informative : features that actually affect output
# n_redundant   : correlated / duplicate features
# n_classes     : binary classification (0 or 1)

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# --------------------------------------------------
# STEP 2: Split Dataset into Training and Testing Sets
# --------------------------------------------------
# 70% training, 30% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# --------------------------------------------------
# STEP 3: Train Support Vector Machine (Hinge Loss)
# --------------------------------------------------
# LinearSVC:
# - Uses hinge loss
# - Maximizes margin between classes
# - Does NOT output probabilities

svm_model = LinearSVC(
    max_iter=5000,
    random_state=42
)

svm_model.fit(X_train, y_train)

# --------------------------------------------------
# STEP 4: Train Logistic Regression (Logistic Loss)
# --------------------------------------------------
# Logistic Regression:
# - Uses log-loss (cross-entropy loss)
# - Maximizes likelihood
# - Outputs probabilities

log_model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

log_model.fit(X_train, y_train)

# --------------------------------------------------
# STEP 5: Evaluate Models
# --------------------------------------------------

svm_train_acc = svm_model.score(X_train, y_train)
svm_test_acc = svm_model.score(X_test, y_test)

log_train_acc = log_model.score(X_train, y_train)
log_test_acc = log_model.score(X_test, y_test)

# --------------------------------------------------
# STEP 6: Display Results Clearly
# --------------------------------------------------

print("Hinge Loss vs Logistic Loss Optimization\n")

print("Support Vector Machine (Hinge Loss):")
print(f"  Training Accuracy : {svm_train_acc:.3f}")
print(f"  Testing Accuracy  : {svm_test_acc:.3f}")
print("  Optimization     : Maximizes margin\n")

print("Logistic Regression (Logistic Loss):")
print(f"  Training Accuracy : {log_train_acc:.3f}")
print(f"  Testing Accuracy  : {log_test_acc:.3f}")
print("  Optimization     : Maximizes likelihood (probabilities)\n")