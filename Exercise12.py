# ================================
# K-MEANS CLUSTERING ON IRIS DATA
# ================================

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------
# 1. Load the Iris dataset
# --------------------------------
iris = load_iris()

# Features (X) -> 4 features:
# Sepal Length, Sepal Width, Petal Length, Petal Width
X = iris.data

# True labels (just for comparison, not used in training)
y = iris.target

# --------------------------------
# 2. Apply K-Means Clustering
# --------------------------------
# We choose 3 clusters because Iris dataset has 3 species
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# --------------------------------
# 3. Get Results
# --------------------------------

# Cluster label for each data point
cluster_labels = kmeans.labels_

# Cluster centers (centroids)
cluster_centers = kmeans.cluster_centers_

print("Cluster Labels for each sample:\n")
print(cluster_labels)

print("\nCluster Centers (Centroids):\n")
print(cluster_centers)

# --------------------------------
# 4. Compare with Actual Labels (Optional)
# --------------------------------
# This is just for understanding how clustering grouped the data
print("\nFirst 20 Actual Labels:")
print(y[:20])

print("\nFirst 20 Cluster Labels:")
print(cluster_labels[:20])

# --------------------------------
# 5. Visualization (2D Projection)
# --------------------------------
# We'll use only first two features for visualization:
# Sepal Length (X[:, 0]) and Sepal Width (X[:, 1])

plt.figure(figsize=(8, 6))

# Plot data points colored by cluster label
plt.scatter(
    X[:, 0], 
    X[:, 1], 
    c=cluster_labels,
    cmap='viridis',
    marker='o',
    edgecolors='black',
    s=100
)

# Plot cluster centers
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    c='red',
    marker='X',
    s=250,
    label="Centroids"
)

plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

plt.show()

# --------------------------------
# 6. Inertia (Cost Function Value)
# --------------------------------
# Inertia = Sum of squared distances of samples to nearest centroid
print("\nK-Means Inertia (WCSS):")
print(kmeans.inertia_)