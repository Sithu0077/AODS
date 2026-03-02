# =========================================
# NAIVE BAYES CLASSIFICATION – IRIS DATASET
# =========================================

# 1. Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load the Iris dataset
iris = load_iris()

# Features (Sepal length, Sepal width, Petal length, Petal width)
X = iris.data

# Target labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)
y = iris.target

# 3. Split dataset into training and testing sets
# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize the Naive Bayes classifier
# GaussianNB is used because features are continuous
nb_model = GaussianNB()

# 5. Train the model using training data
nb_model.fit(X_train, y_train)

# 6. Make predictions on test data
y_pred = nb_model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 8. Print results
print("Accuracy of Naive Bayes Classifier:")
print(f"{accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(conf_matrix, "\n")

print("Classification Report:")
print(class_report)