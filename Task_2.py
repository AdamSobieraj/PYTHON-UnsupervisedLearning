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

# PCA

# Przed standaryzacją
print('Przed standaryzacją')
print('Średnia:\n', X_train.mean(axis=0))
print('Odchylenie standardowe:\n', X_train.std(axis=0))

# Standaryzacja
X_train_scaled = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

print('\nPo standaryzacji')
print('Średnia:\n', X_train_scaled.mean(axis=0))
print('Odchylenie standardowe:\n', X_train_scaled.std(axis=0))

# Macierz kowariancji
covariance_matrix = np.cov(X_train_scaled.T)
print(covariance_matrix)

# Rozkład macierzy kowariancji
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print('Wartości własne:\n', eigenvalues, '\n\nWektory własne:\n', eigenvectors)

explained_variance = [round((i/np.sum(eigenvalues)), 3) for i in sorted(eigenvalues, reverse=True)]
print(explained_variance)

eigenpairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigenpairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigenpairs[0][1][:, np.newaxis],
               eigenpairs[1][1][:, np.newaxis]))
pc1 = X_train_scaled.dot(w.T[0])
pc2 = X_train_scaled.dot(w.T[1])

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
ax.scatter(pc1, pc2, c='black', s=60)
ax.set_xlabel('PC1', rotation=0, loc='center', size=15)
ax.set_ylabel('PC2', rotation=90, loc='center', size=15)
plt.show()
