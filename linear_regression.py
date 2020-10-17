import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf

"""
Name: Leslie Le
Date: 10/16/2020

linear_regression.py

v1..Linear Regression between LOR and CGPA
....LOSS: 0.77243114
....WEIGHT: 1.2640988
....BIAS: 4.0585027
"""

class LinearRegression():
    def __init__(self):
        self.w0 = tf.Variable(np.random.randn(), name='weight')
        self.bias = tf.Variable(np.random.randn(), name='bias')

    def __call__(self, x):
        return self.w0 * x + self.bias

def loss(y, pred):
    return tf.reduce_mean(tf.square(y - pred))

def train(model, x, y, lr=0.006):
    with tf.GradientTape() as t:
        modelLoss = loss(y, model(x))
        change = t.gradient(modelLoss, [model.w0, model.bias])
    model.w0.assign_sub(change[0]*lr) # change weight with first gradient given
    model.bias.assign_sub(change[1]*lr) # change bias with second gradient

df = pd.read_csv("Admission_Predict.csv", header=0,
                 usecols=[5,6,8])
df.dropna(inplace=True)
print(df)

x = tf.cast(df['LOR '].values, tf.float32)
y = tf.cast(df['CGPA'].values, tf.float32)

linear = LinearRegression()

epochs = 1000
for epoch in range(epochs):
  modelLoss = loss(y, linear(x))
  train(linear, x, y)
  finalWeight = linear.w0
  finalBias = linear.bias

print("LOSS: " + str(modelLoss.numpy()))
print("WEIGHT: " + str(linear.w0.numpy()))
print("BIAS: " + str(linear.bias.numpy()))


for i in range(len(x)):

    if (df.iloc[i]['Chance of Admit '] == 1):
        plt.plot(x[i],y[i], 'x', color='red', label='Admitted')
    else:
        plt.plot(x[i],y[i], 'o', color='green', label='not admitted')

plt.plot(x, finalBias + finalWeight*x, color='blue')

#axis
plt.ylabel('age')
plt.xlabel('time')

plt.show()
plt.show()

