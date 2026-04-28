from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=8, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))