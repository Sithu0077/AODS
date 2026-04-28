import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
df = pd.DataFrame({
    'F1': np.random.rand(100),
    'F2': np.random.rand(100),
    'F3': np.random.rand(100),
    'F4': np.random.rand(100)
})

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
result = pca.fit_transform(scaled)

print("Explained Variance:", pca.explained_variance_ratio_)

# Plot
plt.scatter(result[:,0], result[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Result")
plt.show()