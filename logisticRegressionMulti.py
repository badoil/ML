#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:48:47 2023

@author: oneal.oh
"""

import math as m
import numpy as np
import matplotlib.pylab as plt


plt.close("all")

N = 100
d1 = np.random.multivariate_normal(mean=[0, 2], cov=[[2, -3], [-3, 5]], size = N)
d2 = np.random.multivariate_normal(mean=[8, 6], cov=[[5, -3], [-3, 8]], size = N)

plt.scatter(d1[:, 0], d1[:, 1], c="b")
plt.scatter(d2[:, 0], d2[:, 1], c="r")


x1 = np.c_[np.ones([N,1]), d1]
x2 = np.c_[np.ones([N,1]), d2]
X = np.r_[x1, x2]
y1 = np.zeros([N, 1])
y2 = np.ones([N, 1])
y = np.r_[y1, y2]


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


eta = 0.1
n_iterations = 100
wGD = np.zeros([3, 1])
wGDbuffer = np.zeros([3, n_iterations+1])

for iteration in range(n_iterations):
    mu = sigmoid(wGD.T.dot(X.T)).T
    gradients = X.T.dot(mu - y)
    wGD = wGD - eta * gradients


# result, test
x1Sig = np.linspace(-5, 10, 100)
x2Sig = np.linspace(-5, 10, 100)
[x1Sig, x2Sig] = np.meshgrid(x1Sig, x2Sig)
ySig = sigmoid(wGD[0] + wGD[1]*x1Sig + wGD[2]*x2Sig)
ax = plt.axes(projection = "3d")
ax.plot_surface(x1Sig, x2Sig, ySig, cmap="viridis")