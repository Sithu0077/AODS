import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ==========================================================
# K-MEANS: Initialization Comparison & Effect of K
# ==========================================================

# ----------------------------------------------------------
# STEP 1: Generate Synthetic Clustered Data
# ----------------------------------------------------------
# n_samples  : total number of data points
# centers    : actual number of clusters in data
# cluster_std: spread (variance) of each cluster
# random_state: ensures reproducibility

X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

print("Dataset generated with 4 true clusters.\n")


# ----------------------------------------------------------
# STEP 2: Compare Initialization Methods
# ----------------------------------------------------------
# K-Means can initialize cluster centers in two main ways:
# 1. "random"      -> randomly choose centroids
# 2. "k-means++"   -> smarter initialization to spread centers apart

print("K-Means Initialization Comparison\n")

for init_method in ["random", "k-means++"]:

    kmeans = KMeans(
        n_clusters=4,     # We know true clusters = 4
        init=init_method,
        n_init=10,        # Run 10 times, pick best solution
        random_state=42
    )

    # Train model
    kmeans.fit(X)

    print(f"Initialization Method : {init_method}")
    print(f"  Inertia             : {kmeans.inertia_:.2f}")
    print(f"  Iterations          : {kmeans.n_iter_}\n")

# ----------------------------------------------------------
# STEP 3: Analyze Effect of Number of Clusters (K)
# ----------------------------------------------------------
# Inertia = Sum of squared distances of samples to nearest centroid
# Lower inertia → tighter clusters
# Increasing K always reduces inertia

print("Effect of Number of Clusters on Inertia\n")

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    kmeans.fit(X)

    print(f"Number of clusters (K) = {k}")
    print(f"  Inertia              = {kmeans.inertia_:.2f}\n")