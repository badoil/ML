# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math as m
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

plt.close("all")

mylist = np.array([1, 2, 3])

print(mylist)

df_load = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_LinearRegression.txt", 
                      sep="\s+")


xxRaw = np.array(df_load.values[:, 0]) 
yyRaw = np.array(df_load.values[:, 1])
plt.plot(xxRaw, yyRaw, "r.")


N = len(xxRaw)
xx_bias = np.c_[np.ones([100, 1]), xxRaw]   # padding
yy = yyRaw.reshape(N, 1)

# using normal equation
wOLS = np.linalg.inv(xx_bias.T.dot(xx_bias)).dot(xx_bias.T).dot(yy)
x_sample = np.linspace(0, 2.0, 101)
x_sample_bias = np.c_[np.ones([101, 1]), x_sample]
y_predict = wOLS.T.dot(x_sample_bias.T)
x_sample_row = x_sample.reshape(1, 101)
# plt.plot(x_sample_row, y_predict, "b.-")
# plt.show()


# using gradient descent
eta = 0.1
NIteration = 100
wGD = np.array([0, 0]).reshape(2, 1)

for iteration in range(NIteration):
        grad = -(2/N)*(xx_bias.T.dot(yy-xx_bias.dot(wGD)))
        wGD = wGD - grad * eta
        print(wGD)
        yGD_pred = wGD.T.dot(x_sample_bias.T)
        plt.plot(x_sample_row, yGD_pred, "b.-")
        
