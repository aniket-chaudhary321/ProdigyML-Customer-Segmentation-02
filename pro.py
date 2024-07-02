import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the dataset into a Pandas DataFrame
df = pd.read_csv('Mall_Customers.csv')

# Selecting the features for clustering
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Splitting the data into training and testing sets
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train_scaled)  # Fit on training data
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-cluster sum of squares
plt.show()

# Based on the elbow method, choose the optimal number of clusters
optimal_clusters = 5

# Apply K-means clustering with the optimal number of clusters on training data
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
train_cluster_labels = kmeans.fit_predict(X_train_scaled)

# Apply K-means clustering to test data using the same model
test_cluster_labels = kmeans.predict(X_test_scaled)

# Add cluster labels to the original data based on the index of X_train
X_train['Cluster'] = train_cluster_labels

# Merge X_train back to df to add cluster labels
df = pd.merge(df, X_train['Cluster'], left_index=True, right_index=True, how='left')

# Visualize the clusters (assuming 3D plot for Age, Annual Income, and Spending Score)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['Age'], X_train['Annual Income (k$)'], X_train['Spending Score (1-100)'],
           c=train_cluster_labels, cmap='viridis', s=60)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title("Customer Segmentation", fontsize=15)
plt.show()

# Pairplot for selected features colored by cluster
sns.pairplot(df, vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], hue='Cluster', palette='viridis', diag_kind='kde')
plt.suptitle('Pairplot for Customer Segmentation')
plt.show()

# Box plots for each feature by cluster
for feature in features.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y=feature, data=df, palette='viridis')
    plt.title(f'Cluster-wise Distribution of {feature}')
    plt.show()

# Silhouette plot for evaluating cluster quality
silhouette_vals = silhouette_samples(X_train_scaled, train_cluster_labels)
plt.figure(figsize=(10, 6))
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(sorted(df['Cluster'].unique())):
    cluster_silhouette_vals = silhouette_vals[train_cluster_labels == cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    color = plt.cm.viridis(float(i) / len(df['Cluster'].unique()))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(y_ticks, sorted(df['Cluster'].unique()))
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient values')
plt.title('Silhouette Plot for Clusters')
plt.show()

# Compute silhouette scores
silhouette_avg = silhouette_score(X_train_scaled, train_cluster_labels)
print(f'Average Silhouette Score: {silhouette_avg:.2f}')

# Print silhouette score for each cluster (optional)
for i in range(optimal_clusters):
    cluster_silhouette_vals = silhouette_vals[train_cluster_labels == i]
    print(f'Silhouette Score for Cluster {i}: {np.mean(cluster_silhouette_vals):.2f}')
