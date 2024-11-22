import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Set the base path
base_path = 'data/'

# Load data
train_file = os.path.join(base_path, 'samsung_train.txt')
test_file = os.path.join(base_path, 'samsung_test.txt')

X_train = np.loadtxt(train_file)
X_test = np.loadtxt(test_file)

# Load labels
train_labels_file = os.path.join(base_path, 'samsung_train_labels.txt')
test_labels_file = os.path.join(base_path, 'samsung_test_labels.txt')

y_train = np.loadtxt(train_labels_file)
y_test = np.loadtxt(test_labels_file)

# Scale the data
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Determine optimal number of clusters using elbow method
distortions = []
inertias = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_X_train)
    distortions.append(kmeans.inertia_)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# Choose optimal number of clusters (e.g., k=5 for this example)
optimal_k = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_X_train)
labels = kmeans.labels_

# Visualize results
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue', 'yellow', 'purple'][:optimal_k]
for i in range(optimal_k):
    plt.scatter(X_train[:, 0], X_train[:, 1], c=[colors[j] if j == i else 'gray' for j in labels], alpha=0.8)

    # Plot centroids
    center = kmeans.cluster_centers_[i]
    plt.scatter(center[0], center[1], marker='*', s=200, c=colors[i], label=f'Cluster {i + 1}')

plt.title('K-means Clustering Results')
plt.legend()
plt.show()

# Evaluate clustering quality
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(scaled_X_train, labels)
print(f"Silhouette Score: {silhouette}")

# Predict labels for test data
test_labels = kmeans.predict(scaled_X_test)

# Evaluate model on test data
accuracy = sum(test_labels == y_test) / len(y_test)
print(f"Accuracy on test data: {accuracy:.2f}")

# Compare with true labels
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='red',
            label='Predicted Centroids')
plt.title('Comparison with True Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
