"""
Testing a DBSCAN unsupervised anomaly detection scenario with the following order:
1 - Sample database loaded from here: https://zenodo.org/record/7006575
2 - StandardScaler is employed to scale the dataset
3 - epsilon is found based on the assumption that training data belongs to normal machine
4 - DBSCAN is used to define normal machine operating states (dense clusters)
5 - Points labeled -1 by DBSCAN are automatically treated as noise/anomalies
6 - For each new vibration vector, if it falls in a noise region it is flagged as an anomaly
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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

# Step 2: Scale the training data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)


# Step 3: Find optimal eps using the 0.1 % threshold
# Recommended starting point: min_samples = n_features + 1 or 5
min_samples = len(names)
epsilons = []
n_anomalys = []
for eps in np.arange(1, 100):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    train_labels = dbscan.fit_predict(df_scaled)
    train_is_anomaly = train_labels == -1
    n_clusters_found = len(set(train_labels)) - \
        (1 if -1 in train_labels else 0)
    n_anomaly = np.sum(train_is_anomaly)
    anomaly_threshold = 0.001
    epsilons.append(eps)
    n_anomalys.append(n_anomaly)
    print('eps', eps, 'n_anomaly', n_anomaly)
    if n_anomaly < int(anomaly_threshold * len(df_scaled)):
        break

print(f"\nDBSCAN Results on Training Data:")
print(f"  eps={eps}, min_samples={min_samples}")
print(f"  Number of clusters found: {n_clusters_found}")
print(f"  Training anomalies (noise points): {np.sum(train_is_anomaly)} "
      f"({100*np.mean(train_is_anomaly):.2f}%)")

# Step 4: Load and scale the complete dataset
df_complete = pd.DataFrame(total_data, columns=names)
df_complete_scaled = scaler.transform(df_complete)
df_complete_scaled = pd.DataFrame(
    df_complete_scaled, columns=df_complete.columns)


def predict_dbscan(df_scaled, new_data, eps):
    nbrs = NearestNeighbors(n_neighbors=1).fit(df_scaled)
    distances, _ = nbrs.kneighbors(new_data)
    predicted_labels = []
    for dist in distances:
        if dist > eps:
            predicted_labels.append(-1)   # Anomaly / noise
        else:
            predicted_labels.append(0)
    return np.array(predicted_labels)


# Step 5:Predict anomalies on the complete dataset
# Strategy: for each new point, find its nearest neighbor in training data.
# If the nearest training neighbor is a noise point OR the distance > eps,
# then the new point is flagged as an anomaly.
labels_complete = predict_dbscan(
    df_scaled, df_complete_scaled.values, eps)

# Step 6: Find anomalies in complete dataset
is_anomaly = labels_complete == -1
labels_complete_plot = labels_complete.copy().astype(float)

# Print results
print(f"\nDBSCAN Results on Complete Dataset:")
print(f"Total samples: {len(labels_complete)}")
print(f"Total anomalies detected: {np.sum(is_anomaly)} "
      f"({100*np.sum(is_anomaly)/len(is_anomaly):.2f}%)")
print(f"Total normal samples: {np.sum(~is_anomaly)} "
      f"({100*np.sum(~is_anomaly)/len(is_anomaly):.2f}%)\n")

# Plot results
plt.figure(figsize=(14, 10))

# Plot 1: K-Distance Graph (replaces KMeans Elbow Method)
plt.subplot(2, 2, 1)
plt.plot(epsilons, n_anomalys)
plt.axhline(y=f'{100*anomaly_threshold}%', color='r', linestyle='--', linewidth=2,
            label=f'Chosen eps={eps}')
plt.xlabel("epsilon distance")
plt.ylabel("Anomalies in training data")
plt.title(f'min_samples={min_samples}')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training cluster label distribution
plt.subplot(2, 2, 2)

# Plot 3: Predicted states over time
plt.subplot(2, 2, 3)
plt.plot(labels_complete_plot, linewidth=0.5)
plt.axhline(y=-0.7, color='r', linestyle='--', label='Anomaly marker (-1)')
plt.xlabel("Sample Index")
plt.ylabel("Predicted Cluster State")
plt.title(
    f"DBSCAN Predicted States")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Plot 4: Anomaly flags over time (binary view)
plt.subplot(2, 2, 4)
plt.plot(is_anomaly.astype(int), linewidth=0.5, alpha=0.7, color='red')
plt.xlabel("Sample Index")
plt.ylabel("Anomaly (1 = Yes, 0 = No)")
plt.title("Anomaly Flags Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
