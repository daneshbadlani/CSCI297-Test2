"""
Sam Bluestone
Test 2
Exploratory data analysis for the admissions dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.preprocessing import OneHotEncoder
import sys

#read the data into a pandas dataframe
df = pd.read_csv('Admission_Predict.csv')
# df.info()
df.dropna(inplace=True)
df_ohe = pd.get_dummies(df['Race'])
df = df[[i for i in list(df.columns) if i not in ['Serial No.', 'Race']]]
df.dropna(inplace=True)
for race, column in zip(['Asian', 'african american', 'latinx', 'white'], df_ohe.columns):
    df.insert(len(df.columns), race, df_ohe[race])

df.columns = ["GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "CoA", "SES", "ASIAN","AA","LAT","WHITE"]

cols = ["GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "CoA", "SES", "ASIAN","AA","LAT","WHITE"]


#create the scatter plot matrix showing the plots of each feature against each other
scatterplotmatrix(df[df.columns].values, figsize=(20, 16), 
                  names=cols, alpha=0.5)
plt.tight_layout()
plt.savefig("scatterplot_matrix.png")
plt.show()

#create a heatmap with all of the correlation coefficients to determine how correlated a given pair of features are
cm = np.corrcoef(df[df.columns].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols, figsize=(10, 10))
plt.savefig("corr_matrix.png")
plt.show()
