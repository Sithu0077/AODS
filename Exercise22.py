from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = LogisticRegression(max_iter=1000)

# RFE
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X_train, y_train)

print("Selected Features:", rfe.support_)
print("Feature Ranking:", rfe.ranking_)