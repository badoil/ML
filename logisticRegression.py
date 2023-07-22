#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:17:13 2023

@author: oneal.oh
"""

import math as m
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

plt.close("all")

mylist = np.array([1, 2, 3])

print(mylist)

df_load = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_LogisticRegression.txt", 
                      sep="\s+")

xxRaw = np.array(df_load.values[:, 0]) 
yyRaw = np.array(df_load.values[:, 1])
#plt.plot(xxRaw, yyRaw, "r.")

def sigmoid(x):
    return 1.0/(1+np.exp(-x))


N = len(xxRaw)
x_bias = np.c_[np.ones([N, 1]), xxRaw].T
y = yyRaw.reshape(N,1)
X = x_bias.T

eta = 0.1
n_iterations = 1000
wGD = np.zeros([2, 1])
WGDbuffer = np.zeros([2, n_iterations + 1])
mu = sigmoid(wGD.T.dot(x_bias)).T

for iteration in range(n_iterations):
    mu = sigmoid(wGD.T.dot(x_bias)).T
    gradients = X.T.dot(mu-y)
    wGD = wGD - eta*gradients
#    WGDbuffer[:, iteration+1] = [wGD[0], wGD[1]]


xxTest = np.linspace(0, 10, num=N).reshape(N, 1)
xxTest_bias = np.c_[np.ones([N, 1]), xxTest]
aa = sigmoid(wGD.T.dot(xxTest_bias.T))

plt.plot(xxTest, sigmoid(wGD.T.dot(xxTest_bias.T)).T, "r-.")
