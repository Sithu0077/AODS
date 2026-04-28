import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

# Split labeled (10%) and unlabeled (90%)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.9, random_state=42)

# Initial model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_labeled, y_labeled)

# Self-training
for i in range(5):
    probs = model.predict_proba(X_unlabeled)
    preds = model.predict(X_unlabeled)
    confidence = np.max(probs, axis=1)

    high_conf = np.where(confidence > 0.9)[0]

    if len(high_conf) == 0:
        print("No confident samples, stopping early")
        break

    X_new = X_unlabeled[high_conf]
    y_new = preds[high_conf]

    X_labeled = np.vstack((X_labeled, X_new))
    y_labeled = np.concatenate((y_labeled, y_new))

    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[high_conf] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]

    model.fit(X_labeled, y_labeled)
    print(f"Iteration {i+1}, Labeled size: {len(y_labeled)}")

# Final accuracy
if len(X_unlabeled) > 0:
    y_pred = model.predict(X_unlabeled)
    print("Final Accuracy:", accuracy_score(y_unlabeled, y_pred))