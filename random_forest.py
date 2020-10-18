"""
Name: Danesh Badlani
Test 2
Implementation of Random Forest
"""

# import statements
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#read the data into a pandas dataframe
df = pd.read_csv('Admission_Predict.csv')
df.dropna(inplace=True)


df.columns = ["No", "GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "CoA", "RACE", "SES"]


#print(df.describe())
X = df[['GRE', 'TOEFL', 'CGPA']]
y = df['CoA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# chose 0.82 because it is the 3rd quartile for chance of admit
ty_train=[1 if CoA > 0.82 else 0 for CoA in y_train] # learned from internet
ty_train=np.array(ty_train)

ty_test=[1 if CoA > 0.82 else 0 for CoA in y_test] #learned from internet
ty_test=np.array(ty_test)

forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, ty_train)
rf_pred = forest.predict(X_test)
print("Random Forest Accuracy: %.3f" % accuracy_score(ty_test, rf_pred))
print("Random Forest F1-Score: %.3f" % f1_score(ty_test, rf_pred))
print("Random Forest Precision: %.3f" % precision_score(ty_test, rf_pred))
print("Random Forest Recall: %.3f" % recall_score(ty_test, rf_pred))
# Random Forest Accuracy: 0.944
# Random Forest F1-Score: 0.920
# Random Forest Precision: 0.958
# Random Forest Recall: 0.885