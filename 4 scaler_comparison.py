import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

names = ['meanX_F', 'meanY_F', 'meanZ_F',
         'stdX_F', 'stdY_F', 'stdZ_F',
         'shapeX_F', 'shapeY_F', 'shapeZ_F',
         'rmsX_F', 'rmsY_F', 'rmsZ_F',
         'ImpulseX_F', 'ImpuleY_F', 'ImpulseZ_F',
         'ppX_F', 'ppY_F', 'ppZ_F',
         'KurtX_F', 'KurtY_F', 'KurtZ_F',
         'CrestX_F', 'CrestY_F', 'CrestZ_F',
         'SkewX_F', 'SkewY_F', 'SkewZ_F']
metrics = np.genfromtxt(f'data/feature_VBL-VA001.csv', delimiter=',')
df = pd.DataFrame(metrics, columns=names)

# Apply scalers
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

df_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
df_mm = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)

for feature in df.columns.tolist():
    # Choose a feature to visualize
    # feature = df_features.columns[0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    # Original
    axes[0].hist(df[feature], bins=50)
    axes[0].set_title(f"Original - {feature}")
    # StandardScaler
    axes[1].hist(df_std[feature], bins=50)
    axes[1].set_title(f"StandardScaler - {feature}")
    # MinMaxScaler
    axes[2].hist(df_mm[feature], bins=50)
    axes[2].set_title(f"MinMaxScaler - {feature}")
    plt.tight_layout()
plt.show()

