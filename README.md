# AODS
A machine learning project demonstrating best practices using Scikit-learn Pipelines, Ridge Regression (L2 regularization), and 5-fold cross-validation. Features proper scaling, data leakage prevention, and robust model evaluation on the built-in Diabetes dataset.

🧠 Machine Learning Pipeline Optimization using Ridge Regression
📌 Overview

This project demonstrates a professional machine learning workflow using Scikit-learn Pipeline, Ridge Regression (L2 Regularization), and 5-Fold Cross-Validation on the built-in Diabetes dataset.

The goal is to showcase best practices in model development, including:

Proper feature scaling

Regularization to prevent overfitting

Cross-validation for robust performance evaluation

Prevention of data leakage using pipelines

🚀 Key Features

✅ End-to-end ML pipeline using Pipeline

✅ Feature scaling with StandardScaler

✅ Ridge Regression (L2 regularization)

✅ 5-Fold Cross-Validation

✅ R² performance evaluation

✅ Clean and reproducible workflow

✅ No internet dependency (uses built-in dataset)

🛠 Technologies Used

Python

NumPy

Scikit-learn

⚙️ How It Works
1️⃣ Data Loading

The built-in Diabetes dataset is loaded using load_diabetes().

2️⃣ Pipeline Construction

A Scikit-learn Pipeline ensures:

Standardization is applied correctly

No data leakage during cross-validation

Clean, modular workflow

Pipeline steps:

StandardScaler() – Feature normalization

Ridge(alpha=1.0) – Regularized linear regression

3️⃣ Cross-Validation

The model is evaluated using 5-fold cross-validation with R² scoring to ensure reliable generalization performance.

📊 Output

Individual fold R² scores

Average cross-validation score

Clear interpretation of results

🎯 Why This Project Matters

This project highlights three essential ML engineering principles:

Scaling stabilizes optimization

Regularization controls model complexity

Cross-validation ensures generalization

It reflects real-world best practices used in production ML systems.

🧩 Concepts Demonstrated

Bias-Variance Tradeoff

L2 Regularization

Model Generalization

Data Leakage Prevention

Cross-Validation Techniques

▶️ How to Run
pip install scikit-learn numpy
python your_script_name.py


📈 Future Improvements

Hyperparameter tuning using GridSearchCV

Comparison with Lasso Regression

Validation curve visualization

Nested cross-validation
