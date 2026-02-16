import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

names = ['meanX_F', 'meanY_F', 'meanZ_F',
         'stdX_F', 'stdY_F', 'stdZ_F',
         'shapeX_F', 'shapeY_F', 'shapeZ_F',
         'rmsX_F', 'rmsY_F', 'rmsZ_F',
         'ImpulseX_F', 'ImpuleY_F', 'ImpulseZ_F',
         'ppX_F', 'ppY_F', 'ppZ_F',
         'KurtX_F', 'KurtY_F', 'KurtZ_F',
         'CrestX_F', 'CrestY_F', 'CrestZ_F',
         'SkewX_F', 'SkewY_F', 'SkewZ_F', 'label']
metrics = np.genfromtxt(f'data/feature_VBL-VA001.csv', delimiter=',')
label_def = {0: 'normal', 1: 'unbalance',
             2: 'misalignment', 3: 'bearing fault'}
labels = np.genfromtxt(f'data/label_VBL-VA001.csv')
total_data = np.hstack([metrics, labels.reshape(-1, 1)])
df = pd.DataFrame(total_data, columns=names)
df_correlation_check_X = df[['shapeX_F', 'shapeY_F', 'shapeZ_F',
                             'rmsX_F', 'rmsY_F', 'rmsZ_F',
                             'ppX_F', 'ppY_F', 'ppZ_F',
                             'KurtX_F', 'KurtY_F', 'KurtZ_F',
                             'CrestX_F', 'CrestY_F', 'CrestZ_F']]
corr1 = df_correlation_check_X.corr()
high_corr = corr1.unstack().dropna()
high_corr = high_corr[(abs(high_corr) > 0.75) & (abs(high_corr) < 1)]
print(high_corr.sort_values(ascending=False))
sb.heatmap(corr1, xticklabels=corr1.columns, yticklabels=corr1.columns,
           vmin=-1, vmax=1)
sb.pairplot(df_correlation_check_X)  # , hue='label')
plt.show()
