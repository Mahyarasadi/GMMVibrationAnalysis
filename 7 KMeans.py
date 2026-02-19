"""
Testing a KMeans unsupervised anomaly detection scenario with the following order:
1 - Sample database loaded from here: https://zenodo.org/record/7006575
2 - StandardScaler is employed to scale the dataset
3 - KMeans is used to define normal machine operating states (clusters)
4 - The machine operation is classified into a few operating states
5 - Distance to the nearest cluster center is used as the anomaly score
6 - For each new vibration vector, if the distance exceeds the threshold it is flagged as an anomaly
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

names = ['meanX_F', 'meanY_F', 'meanZ_F',
         'stdX_F', 'stdY_F', 'stdZ_F',
         'shapeX_F', 'shapeY_F', 'shapeZ_F',
         'rmsX_F', 'rmsY_F', 'rmsZ_F',
         'ImpulseX_F', 'ImpuleY_F', 'ImpulseZ_F',
         'ppX_F', 'ppY_F', 'ppZ_F',
         'KurtX_F', 'KurtY_F', 'KurtZ_F',
         'CrestX_F', 'CrestY_F', 'CrestZ_F',
         'SkewX_F', 'SkewY_F', 'SkewZ_F']

total_data = np.genfromtxt(f'data/feature_VBL-VA001.csv', delimiter=',')
df = pd.DataFrame(total_data[0:1000, :], columns=names)

# Step 1: Scale the training data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Step 2: Find optimal number of clusters using the Elbow Method (inertia)
inertias = []
ks = range(1, 20)

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)

# Choose k with elbow point
best_k = 10

# Step 3: Fit KMeans with best_k and compute distances to cluster centers
kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
kmeans.fit(df_scaled)

def compute_distances(data, model):
    """Compute each sample's distance to its nearest cluster center."""
    centers = model.cluster_centers_
    labels = model.predict(data)
    distances = np.linalg.norm(data - centers[labels], axis=1)
    return distances

train_distances = compute_distances(df_scaled.values, kmeans)

# Step 4: Set a threshold based on the high percentile of training distances
threshold_percentile = 99.9  # Flag top 1% most distant points as anomalies
distance_threshold = np.percentile(train_distances, threshold_percentile)

# Step 5: Load and scale the complete dataset
df_complete = pd.DataFrame(total_data, columns=names)
df_complete_scaled = scaler.transform(df_complete)
df_complete_scaled = pd.DataFrame(df_complete_scaled, columns=df_complete.columns)

# Step 6: Predict states & distances for all data
labels_complete = kmeans.predict(df_complete_scaled)
complete_distances = compute_distances(df_complete_scaled.values, kmeans)

# Step 7: Find anomalies
is_anomaly = complete_distances > distance_threshold
labels_complete_plot = labels_complete.copy().astype(float)
labels_complete_plot[is_anomaly] = -1  # Mark anomalies as -1

# Print results
print(f"\nDistance threshold (at {threshold_percentile}th percentile): {distance_threshold:.4f}")
print(f"Total samples: {len(labels_complete)}")
print(f"Total anomalies detected: {np.sum(is_anomaly)} ({100*np.sum(is_anomaly)/len(is_anomaly):.2f}%)")
print(f"Total normal samples: {np.sum(~is_anomaly)} ({100*np.sum(~is_anomaly)/len(is_anomaly):.2f}%)\n")

# Plot results
plt.subplot(2, 2, 1)
plt.plot(ks, inertias)
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title('Elbow Method: Inertia vs Number of Clusters')
plt.grid('major')

plt.subplot(2, 2, 2)
plt.hist(train_distances, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(distance_threshold, color='r', linestyle='--', linewidth=2,
            label=f'Threshold ({threshold_percentile}th percentile)')
plt.xlabel('Distance to Cluster Center')
plt.ylabel('Frequency')
plt.title('Distribution of Distances on Training Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.plot(labels_complete_plot, linewidth=0.5)
plt.axhline(y=-0.7, color='r', linestyle='--', label='Anomaly marker')
plt.xlabel("Sample Index")
plt.ylabel("Predicted State")
plt.title(f"KMeans Predicted States for Complete Dataset (k={best_k})")
plt.yticks(range(best_k))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.plot(complete_distances, linewidth=0.5, alpha=0.7)
plt.axhline(y=distance_threshold, color='r', linestyle='--', linewidth=2,
            label=f'Threshold ({threshold_percentile}th percentile)')
plt.xlabel("Sample Index")
plt.ylabel("Distance to Cluster Center")
plt.title("Anomaly Scores (Distances) Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()