from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = load_iris()
X = data.data
y = data.target

model = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_acc = []

for train, test in kf.split(X):
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    kf_acc.append(accuracy_score(y[test], pred))

print("K-Fold Accuracy:", np.mean(kf_acc))

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_acc = []

for train, test in skf.split(X, y):
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    skf_acc.append(accuracy_score(y[test], pred))

print("Stratified K-Fold Accuracy:", np.mean(skf_acc))

# Leave One Out
loo = LeaveOneOut()
loo_acc = []

for train, test in loo.split(X):
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    loo_acc.append(accuracy_score(y[test], pred))

print("LOO Accuracy:", np.mean(loo_acc))