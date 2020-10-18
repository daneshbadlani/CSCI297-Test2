"""
Sam Bluestone
Test 2
Implementation of a Naive Bayes Classifier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection


#read the data into a pandas dataframe
df = pd.read_csv('Admission_Predict.csv')
df.dropna(inplace=True)


df.columns = ["No", "GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "CoA", "RACE", "SES"]


X = df[['GRE', 'TOEFL', 'CGPA']]
y = df['CoA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# chose 0.82 because it is the 3rd quartile for chance of admit
ty_train=[1 if CoA > 0.82 else 0 for CoA in y_train] # learned from internet
ty_train=np.array(ty_train)

ty_test=[1 if CoA > 0.82 else 0 for CoA in y_test] #learned from internet
ty_test=np.array(ty_test)

nb = GaussianNB()
nb.fit(X_train_std, ty_train)
y_pred = nb.predict(X_test_std)


#Model Performance
# #setting performance parameters
kfold = model_selection.KFold(n_splits=10)
#calling the cross validation function
cv_results = model_selection.cross_val_score(GaussianNB(), X_train_std, ty_train, cv=kfold, scoring='accuracy')
#displaying the mean and standard deviation of the prediction
print("%s: %f %s: (%f)" % ('Naive Bayes accuracy', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))
