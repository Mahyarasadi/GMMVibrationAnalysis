import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
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
         'SkewX_F', 'SkewY_F', 'SkewZ_F', 'label']
metrics = np.genfromtxt(f'data/feature_VBL-VA001.csv', delimiter=',')
label_def = {0: 'normal', 1: 'unbalance',
             2: 'misalignment', 3: 'bearing fault'}
labels = np.genfromtxt(f'data/label_VBL-VA001.csv')
total_data = np.hstack([metrics, labels.reshape(-1, 1)])
df = pd.DataFrame(total_data, columns=names)
# x = df[['shapeX_F', 'shapeY_F', 'shapeZ_F',
#                              'rmsX_F', 'rmsY_F', 'rmsZ_F',
#                              'ppX_F', 'ppY_F', 'ppZ_F',
#                              'KurtX_F', 'KurtY_F', 'KurtZ_F',
#                              'CrestX_F', 'CrestY_F', 'CrestZ_F']]
x = df[names]
y = df['label'].to_numpy()

# SVM
c_svm = np.arange(1, 100)
test_accuracy = []#np.empty(len(c_svm))
for i, k in enumerate(c_svm):
    # Setup a knn classifier with c_svm
    clf_svm = SVC(C=k)
    # Do 5-cv to the model
    scores = cross_val_score(clf_svm, x, y, cv=5)
    test_accuracy.append(np.mean(scores))
plt.subplot(3,1,1)
plt.plot(c_svm, test_accuracy, label='SVM Accuracy vs. c')
plt.legend()
plt.grid('major')
plt.xlabel('c')
plt.ylabel('Accuracy')

# KNN
neighbors = np.arange(1, 100)
test_accuracy = []
for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf_knn, x, y, cv=5)
    test_accuracy.append(np.mean(scores))
plt.subplot(3,1,2)
plt.plot(c_svm, test_accuracy, label='KNN Accuracy vs. NNeigbours')
plt.legend()
plt.grid('major')
plt.xlabel('c')
plt.ylabel('Accuracy')
# GNB
var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]

test_accuracy = []
for i, k in enumerate(var_gnb):
    # Setup a knn classifier with k neighbors
    clf_gnb = GaussianNB(var_smoothing=k)
    scores = cross_val_score(clf_gnb, x, y, cv=5)
    test_accuracy.append(np.mean(scores))

plt.subplot(3,1,3)
plt.plot(var_gnb, test_accuracy, label='GNB Accuracy vs. smoothingvar')
plt.legend()
plt.grid('major')
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.show()
