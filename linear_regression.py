"""
Name: Leslie Le
Test 2
Implementation of Linear Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import tensorflow as tf


df = pd.read_csv('Admission_Predict.csv')
df.dropna(inplace=True)

df.columns = ["No", "GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "CoA", "RACE", "SES"]

X = df[['GRE', 'TOEFL', 'CGPA']]
y = df['CoA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)
print('Slope: %.3f' % lin_reg.coef_[0])
print('Intercept: %.3f' % lin_reg.intercept_)
print("R^2 Score: %.3f" % r2_score(lin_pred, y_test))
print("MSE: %.3f" % mean_absolute_error(lin_pred, y_test))
# Slope: 0.002
# Intercept: -1.606
# R^2 Score: 0.776
# MSE: 0.040