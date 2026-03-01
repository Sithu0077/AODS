import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==================================================
# COMPARATIVE STUDY: HINGE LOSS vs LOGISTIC LOSS
# ==================================================

# --------------------------------------------------
# STEP 1: Create Binary Classification Dataset
# --------------------------------------------------
# n_samples     : number of data points
# n_features    : total features
# n_informative : useful features
# n_redundant   : correlated features
# n_classes     : binary classification

X, y = make_classification(
    n_samples=800,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# --------------------------------------------------
# STEP 2: Split into Training and Testing Sets
# --------------------------------------------------
# 70% training, 30% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# --------------------------------------------------
# STEP 3: Train SVM (Hinge Loss)
# --------------------------------------------------
# LinearSVC:
# - Uses hinge loss
# - Maximizes margin
# - Outputs distance from decision boundary

svm = LinearSVC(max_iter=5000, random_state=42)
svm.fit(X_train, y_train)

# --------------------------------------------------
# STEP 4: Train Logistic Regression (Logistic Loss)
# --------------------------------------------------
# Logistic Regression:
# - Uses log-loss
# - Maximizes likelihood
# - Outputs probabilities

log_reg = LogisticRegression(max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)

# --------------------------------------------------
# STEP 5: Evaluate Accuracy
# --------------------------------------------------

svm_acc = svm.score(X_test, y_test)
log_acc = log_reg.score(X_test, y_test)

# --------------------------------------------------
# STEP 6: Compare Decision Behaviour
# --------------------------------------------------

# SVM output: signed distance from decision boundary (margin)
svm_decision = svm.decision_function(X_test)

# Logistic Regression output: probability of class 1
log_prob = log_reg.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# STEP 7: Display Results Clearly
# --------------------------------------------------

print("Comparative Study: Hinge Loss vs Logistic Loss\n")

print("Support Vector Machine (Hinge Loss):")
print(f"  Test Accuracy        : {svm_acc:.3f}")
print("  Decision Output      : Margin values (no probabilities)")
print(f"  Sample Margin Values : {np.round(svm_decision[:5], 3)}\n")

print("Logistic Regression (Logistic Loss):")
print(f"  Test Accuracy        : {log_acc:.3f}")
print("  Decision Output      : Probabilities")
print(f"  Sample Probabilities : {np.round(log_prob[:5], 3)}\n")