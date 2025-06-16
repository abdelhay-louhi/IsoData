import numpy as np
from scipy.spatial.distance import cdist

class ISODATA:
    def __init__(self,
                 K_desired=5,
                 N_min=10,
                 max_iter=100,
                 max_merge_dist=2.0,
                 max_std=1.0,
                 max_split=2,
                 D_factor=1.0,
                 random_state=None):
        self.K_desired = K_desired
        self.N_min = N_min
        self.max_iter = max_iter
        self.max_merge_dist = max_merge_dist  # θ_merge
        self.max_std = max_std                # σ_max
        self.max_split = max_split
        self.D_factor = D_factor              # D1
        self.random_state = np.random.RandomState(random_state)

    def fit(self, X):
        N, dim = X.shape
        k = self.K_desired // 2  # initial number of clusters
        self.centroids = X[self.random_state.choice(N, k, replace=False)]
        for iteration in range(self.max_iter):
            labels = self.assign_clusters(X)
            self.update_centroids(X, labels)
            labels, removed = self.discard_small_clusters(X, labels)
            self.merge_clusters()
            self.split_clusters(X, labels)
            if len(self.centroids) >= self.K_desired:
                break
        self.labels_ = self.assign_clusters(X)

    def assign_clusters(self, X):
        distances = cdist(X, self.centroids)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        new_centroids = []
        for i in range(len(self.centroids)):
            points = X[labels == i]
            if len(points) > 0:
                new_centroids.append(points.mean(axis=0))
        self.centroids = np.array(new_centroids)

    def discard_small_clusters(self, X, labels):
        new_centroids = []
        removed = 0
        for i in range(len(self.centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) <= self.N_min:
                removed += 1
                continue
            new_centroids.append(cluster_points.mean(axis=0))
        self.centroids = np.array(new_centroids)
        new_labels = self.assign_clusters(X)
        return new_labels, removed

    def merge_clusters(self):
        K = len(self.centroids)
        to_merge = []
        for i in range(K):
            for j in range(i+1, K):
                dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                if dist < self.max_merge_dist:
                    to_merge.append((i, j))

        merged = set()
        new_centroids = []
        for i in range(K):
            if i in merged:
                continue
            merged_with = [self.centroids[i]]
            for j in range(i+1, K):
                if (i, j) in to_merge or (j, i) in to_merge:
                    merged_with.append(self.centroids[j])
                    merged.add(j)
            new_centroids.append(np.mean(merged_with, axis=0))
        self.centroids = np.array(new_centroids)

    def split_clusters(self, X, labels):
        new_centroids = []
        num_splits = 0
        overall_avedist = self.compute_overall_avedist(X, labels)

        for i, centroid in enumerate(self.centroids):
            cluster_points = X[labels == i]
            if len(cluster_points) <= 2 * self.N_min + 2:
                new_centroids.append(centroid)
                continue

            std_devs = np.std(cluster_points, axis=0)
            j_max = np.argmax(std_devs)
            if std_devs[j_max] < self.max_std:
                new_centroids.append(centroid)
                continue

            cluster_avedist = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
            if cluster_avedist < self.D_factor * overall_avedist:
                new_centroids.append(centroid)
                continue

            # Perform splitting
            offset = np.zeros_like(centroid)
            offset[j_max] = 1.0
            new_centroids.append(centroid + offset)
            new_centroids.append(centroid - offset)
            num_splits += 1
            if num_splits >= self.max_split:
                break
        self.centroids = np.array(new_centroids)

    def compute_overall_avedist(self, X, labels):
        total = 0
        for i, centroid in enumerate(self.centroids):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            total += np.sum(np.linalg.norm(cluster_points - centroid, axis=1))
        return total / len(X)
