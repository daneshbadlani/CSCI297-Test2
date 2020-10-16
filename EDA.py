"""
Name: Danesh Badlani, Sam Bluestone and Leslie Le
Date: 10/16/2020
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap

# read dataset
df = pd.read_csv("Admission_Predict.csv")
df.dropna(inplace=True)
df = df[[i for i in list(df.columns) if i != "Serial No."]]
df = df.replace(to_replace={"Asian":0, "latinx":1, "african american":2, "white":3})

df.columns = ["GRE", "TOEFL", "UR", "SOP", "LOR", "CGPA", "RES", "COA", "RACE", "SES"]

#create the scatter plot matrix showing the plots of each feature against each other
scatterplotmatrix(df[df.columns].values, figsize=(20, 16), 
                  names=df.columns, alpha=0.5)
plt.tight_layout()
plt.savefig("scatterplot_matrix.png")
plt.show()

#create a heatmap with all of the correlation coefficients to determine how correlated a given pair of features are
cm = np.corrcoef(df[df.columns].values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns, figsize=(5, 5))
plt.savefig("corr_matrix.png")
plt.show()
