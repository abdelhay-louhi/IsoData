from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from isodata_ball_hall import ISODATA

X, y = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)

model = ISODATA(K_desired=5, random_state=42)
model.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='tab10', s=30)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='black', marker='x', s=100)
plt.title("ISODATA Clustering (Ball & Hall, 1965)")
plt.show()
