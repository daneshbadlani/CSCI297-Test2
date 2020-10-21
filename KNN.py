"""
Name: Danesh Badlani
Test 2 
Implementation of KNN Classifier
"""

# import statements
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import export_graphviz

# Graphing
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    classifier.fit(X, y)
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

def knn_multiple():
    for i in [1,2,3,4,5,6,7,8,9,10,25,50,95]:
        knn = KNeighborsClassifier(n_neighbors=i, metric='chebyshev')
        knn.fit(X_train_std, ty_train)
        knn_pred = knn.predict(X_test_std)
        print()
        print("k = %i" %i)
        print("KNN Accuracy: %.3f" % accuracy_score(ty_test, knn_pred))
        print("KNN F1-Score: %.3f" % f1_score(ty_test, knn_pred))
        print("KNN Precision: %.3f" % precision_score(ty_test, knn_pred))
        print("KNN Recall: %.3f" % recall_score(ty_test, knn_pred))

#read the data into a pandas dataframe
df = pd.read_csv('Admission_Predict.csv')
df.dropna(inplace=True)
df.columns = ["No", "GRE", "TOEFL", "UR", "SOP", "LOR",
              "CGPA", "RES", "CoA", "RACE", "SES"]
#print(df.describe())
X = df[['CGPA','GRE', 'TOEFL']]
y = df['CoA']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=1)

# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#X_combined_std = np.vstack((X_train, X_test))
#y_combined = np.hstack((y_train, y_test))

# chose 0.82 because it is the 3rd quartile for chance of admit
ty_train=[1 if CoA > 0.82 else 0 for CoA in y_train] # learned from internet
ty_train=np.array(ty_train)

ty_test=[1 if CoA > 0.82 else 0 for CoA in y_test] #learned from internet
ty_test=np.array(ty_test)

knn = KNeighborsClassifier(n_neighbors=5, metric='chebyshev')
knn.fit(X_train_std, ty_train)
knn_pred = knn.predict(X_test_std)
print("k = 5")
print("KNN Accuracy: %.3f" % accuracy_score(ty_test, knn_pred))
print("KNN F1-Score: %.3f" % f1_score(ty_test, knn_pred))
print("KNN Precision: %.3f" % precision_score(ty_test, knn_pred))
print("KNN Recall: %.3f" % recall_score(ty_test, knn_pred))

X_combined_std = np.vstack((X_train_std[:, 1:], X_test_std[:, 1:]))
y_combined = np.hstack((ty_train, ty_test))
plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105,150))
plt.show()

print("Testing different k values:")
knn_multiple()
""" GRE TOEFL
KNN Accuracy: 0.806
KNN F1-Score: 0.682
KNN Precision: 0.833
KNN Recall: 0.577

GRE CGPA
KNN Accuracy: 0.917
KNN F1-Score: 0.875
KNN Precision: 0.955
KNN Recall: 0.808

TOEFL CGPA
KNN Accuracy: 0.958
KNN F1-Score: 0.941
KNN Precision: 0.960
KNN Recall: 0.923

ALL 3
# KNN Accuracy: 0.917
# KNN F1-Score: 0.870
# KNN Precision: 1.000
# KNN Recall: 0.769
"""
