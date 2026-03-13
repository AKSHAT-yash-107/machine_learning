import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iters):

            # Assign clusters
            clusters = [[] for _ in range(self.k)]

            for idx, point in enumerate(X):
                distances = np.linalg.norm(point - self.centroids, axis=1)
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(idx)

            # Compute new centroids
            new_centroids = np.array([
                X[cluster].mean(axis=0) if cluster else self.centroids[i]
                for i, cluster in enumerate(clusters)
            ])

            # Stop if converged
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return clusters


# ------------------------
# Generate sample dataset
# ------------------------

np.random.seed(42)

cluster1 = np.random.randn(100,2) + np.array([2,2])
cluster2 = np.random.randn(100,2) + np.array([7,7])
cluster3 = np.random.randn(100,2) + np.array([2,8])

X = np.vstack((cluster1, cluster2, cluster3))


# ------------------------
# Run K-Means
# ------------------------

model = KMeans(k=3)
clusters = model.fit(X)
centroids = model.centroids


# ------------------------
# Visualization
# ------------------------

colors = ['red','blue','green']

for i, cluster in enumerate(clusters):
    points = X[cluster]
    plt.scatter(points[:,0], points[:,1], color=colors[i])

plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='X', s=200)
plt.title("K-Means Clustering From Scratch")
plt.show()