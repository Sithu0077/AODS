import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================================
# HYPERPARAMETER OPTIMIZATION:
# Grid Search vs Random Search
# ==========================================================

# ----------------------------------------------------------
# STEP 1: Create Classification Dataset
# ----------------------------------------------------------
# n_samples     : number of samples
# n_features    : total features
# n_informative : useful features
# n_redundant   : correlated features

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

# Split dataset into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# ----------------------------------------------------------
# STEP 2: Define the Model
# ----------------------------------------------------------
model = RandomForestClassifier(random_state=42)

# ----------------------------------------------------------
# STEP 3: Define Hyperparameter Search Space
# ----------------------------------------------------------
# n_estimators : number of trees
# max_depth    : maximum depth of each tree

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}

# ----------------------------------------------------------
# STEP 4: Grid Search (Exhaustive Search)
# ----------------------------------------------------------
# Tries ALL combinations of parameters

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross validation
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

# Evaluate best grid model on test set
grid_best_model = grid_search.best_estimator_
grid_test_acc = accuracy_score(y_test, grid_best_model.predict(X_test))


# ----------------------------------------------------------
# STEP 5: Random Search (Random Sampling)
# ----------------------------------------------------------
# Randomly samples parameter combinations

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=5,                # Only 5 random combinations tested
    cv=5,
    scoring="accuracy",
    random_state=42
)

random_search.fit(X_train, y_train)

# Evaluate best random model on test set
random_best_model = random_search.best_estimator_
random_test_acc = accuracy_score(y_test, random_best_model.predict(X_test))


# ----------------------------------------------------------
# STEP 6: Display Results Clearly
# ----------------------------------------------------------

print("Hyperparameter Optimization Results\n")

print("Grid Search (Exhaustive Search):")
print(f"  Best Parameters : {grid_search.best_params_}")
print(f"  Best CV Score   : {grid_search.best_score_:.3f}")
print(f"  Test Accuracy   : {grid_test_acc:.3f}\n")

print("Random Search (Random Sampling):")
print(f"  Best Parameters : {random_search.best_params_}")
print(f"  Best CV Score   : {random_search.best_score_:.3f}")
print(f"  Test Accuracy   : {random_test_acc:.3f}\n")