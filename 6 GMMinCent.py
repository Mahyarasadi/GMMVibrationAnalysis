"""
Testing a GMM scenario with the following order:
1 -Raw vibration data (The two week training  period) is converted into metrics, like rms, crest, kurtosis, …
2- standard scaler is employed to scale the data set
3- Then all of the new features are fed into an unsupervised singe class (NORMAL) machine state definition method (Here GMM).
4- The machine operation is classified into a few operating states.
5- Log likelihood is to find the GMM covariance score for anomalies
6- Now for each new vibration vector the GMM class is calculated and if the likelihood is lower than the threshold then it is considered as an anomaly

Sample Dataset Webpage:
https://pdm-backend.groundup.ai/?company=15&location=d0e1ae8b-aee0-4621-8529-ad9c221bd6ec&machine=e007477b-a666-4c6a-b63e-aa0031f0c00b&machine_element=1693a3f1-f94a-4a55-a88b-fd0cd32c22f3&deployment=1693a3f1-f94a-4a55-a88b-fd0cd32c22f3&direction=1&mode=12&select_timestamp=2025-12-12T20%3A59%3A58&start_date=2025-03-19&end_date=2026-12-31
"""
# TODO: FFT velocity metrics still seem not to be valid.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

df_init = pd.read_csv('data\dataset.csv')
df_init = df_init.drop(columns=['Timestamp'])
for col in df_init.columns:
    if col.startswith('FFT Velocity'):
        df_init = df_init.drop(columns=col)
on_off_thrshld = 0.5
df_init = df_init[df_init["RMS Velocity (mm/s)"] >= 1]

# Step 1: Find optimal number of Gaussians using BIC
scaler = StandardScaler()
training_n = 24 * 7  # one sample per hour for two weeks
df_training = df_init.iloc[0:training_n, :]
df_scaled = scaler.fit_transform(df_training)
df_scaled = pd.DataFrame(df_scaled, columns=df_training.columns)


# Step 2: Find optimal number of Gaussians using BIC
bics = []
ks = range(1, 20)

for k in ks:
    gmm = GaussianMixture(
        n_components=k, covariance_type='full', random_state=0)
    gmm.fit(df_scaled)
    bics.append(gmm.bic(df_scaled))


# Choose k with lowest BIC
best_k = 8

# Step 3: Fit a GMM model and calculate the covariance scores
gmm = GaussianMixture(n_components=best_k,
                      covariance_type='full', random_state=0)
gmm.fit(df_scaled)
train_log_likelihood = gmm.score_samples(df_scaled)

# Step 4: Set a threshold for the normal state minimum log likelihood
threshold_percentile = 0.001  # 1% Percentile
log_likelihood_threshold = np.percentile(
    train_log_likelihood, threshold_percentile)


# Step 5: Load complete dataset
# Scale complete dataset using the SAME scaler
df_complete_scaled = scaler.transform(df_init)
df_complete_scaled = pd.DataFrame(
    df_complete_scaled, columns=df_init.columns)
print('df_complete_scaled', df_complete_scaled.shape)
print('df_init', df_init.shape)

# Step 6: Predict states & likelihoods for all data using the trained GMM
labels_complete = gmm.predict(df_complete_scaled)
complete_log_likelihood = gmm.score_samples(df_complete_scaled)

# Step 7: Find anomalies
is_anomaly = complete_log_likelihood < log_likelihood_threshold
labels_complete[is_anomaly] = -1  # Mark anomalies as -1

# Print the results
print(
    f"\nLog-likelihood threshold (at {threshold_percentile}th percentile): {log_likelihood_threshold:.4f}")
print(f"Total samples: {len(labels_complete)}")
print(
    f"Total anomalies detected: {np.sum(is_anomaly)} ({100*np.sum(is_anomaly)/len(is_anomaly):.2f}%)")
print(
    f"Total normal samples: {np.sum(~is_anomaly)} ({100*np.sum(~is_anomaly)/len(is_anomaly):.2f}%)\n")

# Plot the results
plt.subplot(2, 2, 1)
plt.plot(ks, bics)
plt.xlabel("k")
plt.grid('major')
plt.ylabel("BIC")
plt.title('BIC vs Number of machine states')

plt.subplot(2, 2, 2)
plt.hist(train_log_likelihood, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(log_likelihood_threshold, color='r', linestyle='--', linewidth=2,
            label=f'Threshold ({threshold_percentile}th percentile)')
plt.xlabel('Log-Likelihood')
plt.ylabel('Frequency')
plt.title('Distribution of Log-Likelihood Scores on Training Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.plot(labels_complete, linewidth=0.5)
plt.axhline(y=-0.7, color='r', linestyle='--', label='Anomaly marker')
plt.xlabel("Sample Index")
plt.ylabel("Predicted State")
plt.title(f"GMM Predicted States for Complete Dataset (k={best_k})")
plt.yticks(range(best_k))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.plot(complete_log_likelihood, linewidth=0.5, alpha=0.7)
plt.axhline(y=log_likelihood_threshold, color='r', linestyle='--', linewidth=2,
            label=f'Threshold ({threshold_percentile}th percentile)')
plt.xlabel("Sample Index")
plt.ylabel("Log-Likelihood")
plt.title("Log-Likelihood Scores Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.figure()
df_init = df_init.reset_index()
plt.plot(df_init.index, df_init["RMS Velocity (mm/s)"], label="RMS Velocity (mm/s)")
plt.show()
