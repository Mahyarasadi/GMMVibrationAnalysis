"""
Testing a GMM scenario with the following order:
1 -Raw vibration data (The two week training  period) is converted into metrics, like rms, crest, kurtosis, …
2- Then a dimensionality reduction procedure is performed on the data using PCA to extract the principal components.
3- Then all of the new features are fed into an unsupervised singe class (NORMAL) machine state definition method (Here GMM).
4- The machine operation is classified into a few operating states.
5- Now for each new vibration vector the GMM class is calculated and if the probability belonging to any operating conditions is lower than a threshold then the machine is called to have an anomaly.

Sample Dataset Webpage:
https://pdm-backend.groundup.ai/?company=15&location=d0e1ae8b-aee0-4621-8529-ad9c221bd6ec&machine=e007477b-a666-4c6a-b63e-aa0031f0c00b&machine_element=1693a3f1-f94a-4a55-a88b-fd0cd32c22f3&deployment=1693a3f1-f94a-4a55-a88b-fd0cd32c22f3&direction=1&mode=12&select_timestamp=2025-12-12T20%3A59%3A58&start_date=2025-03-19&end_date=2026-12-31
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
 
df_init = pd.read_csv('dataset.csv')
on_off_thrshld = 0.5
df_init = df_init[df_init["RMS Velocity (mm/s)"] >= 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled = scaler.fit_transform(df_init.drop(columns=["Timestamp"]))
df_scaled = pd.DataFrame(df_scaled, columns=df_init.drop(columns=["Timestamp"]).columns)
# pca = PCA(n_components=0.95)  # keep 95% of variance
# X_pca = pca.fit_transform(df_scaled)
# print(X_pca)
# Step 3: Find optimal number of Gaussians using BIC
bics = []
ks = range(1, 20)

# for k in ks:
#     gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
#     gmm.fit(df_scaled)
#     bics.append(gmm.bic(df_scaled))
# plt.plot(ks, bics)
# print(ks, bics)
# plt.xlabel("Number of components")
# plt.ylabel("BIC")
# plt.show()

# Choose k with lowest BIC
best_k = 6

# Step 4: Train final GMM
gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=0)
gmm.fit(df_scaled)

# # Step 5: Get cluster labels
labels = gmm.predict(df_scaled)

# Step 6: Add labels to dataframe
df_scaled["cluster"] = labels
# Step 7: Inspect cluster means
# print(df_scaled.groupby("cluster").mean())
print(labels)
