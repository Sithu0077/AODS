import warnings 
warnings.filterwarnings("ignore") 
from sklearn.datasets import load_diabetes 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDRegressor 
import numpy as np 
# -------------------------------------------------- 
# STEP 1: Load and scale dataset 
# -------------------------------------------------- 
X, y = load_diabetes(return_X_y=True) 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# -------------------------------------------------- 
# STEP 2: Centralized Training 
# -------------------------------------------------- 
# Entire dataset is used for training at once 
central_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42) 
central_model.fit(X_scaled, y) 
# -------------------------------------------------- 
# STEP 3: Simulated Distributed Training 
# -------------------------------------------------- 
# Split data into two parts (as if on two machines) 
X_part1, X_part2 = np.array_split(X_scaled, 2) 
y_part1, y_part2 = np.array_split(y, 2) 
# Train separate models on each data partition 
model_1 = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42) 
model_2 = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42) 
model_1.fit(X_part1, y_part1) 
model_2.fit(X_part2, y_part2) 
# -------------------------------------------------- 
# STEP 4: Aggregate model parameters (simple averaging) 
# -------------------------------------------------- 
avg_coef = (model_1.coef_ + model_2.coef_) / 2 
avg_intercept = (model_1.intercept_ + model_2.intercept_) / 2 
# -------------------------------------------------- 
# STEP 5: Display results clearly 
# -------------------------------------------------- 
print("Centralized vs Distributed Training Comparison\n") 
print("Centralized Training:") 
print(f"  Iterations Used : {central_model.n_iter_}") 
print(f"  Model Coeff Mean: {np.mean(central_model.coef_):.3f}\n") 
print("Distributed Training (Simulated):") 
print("  Data split into two parts") 
print(f"  Avg Coeff Mean  : {np.mean(avg_coef):.3f}") 
print("\nNote:") 
print("  Differences occur due to independent optimization and lack of synchronization.") 