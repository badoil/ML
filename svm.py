#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:09:36 2023

@author: oneal.oh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import scipy.stats

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.close("all")

iris = datasets.load_iris()

X = iris["data"][0:100, (2,3)]
Y = iris["target"][0:100]

# f1, ax1 = plt.subplots()
# ax1.plot(X[:,0], X[:,1], "*")

scaler = StandardScaler()
scaler.fit(X)           # 데이터를 표준화하기 위해서 평균, 표준편차 구하기
X_std = scaler.transform(X)   # 구한 평균과 표준편차로 스케일


# SVM 알고리즘 우리가 짤수 있지만 라이브러리를 사용하자

svm_clf = SVC(C=0.01, kernel="linear")
svm_clf.fit(X_std, Y) 

f3, ax3 = plt.subplots()
ax3.plot(X_std[:,0], X_std[:,1], "*")

[x0Min, x0Max] = [min(X_std[:,0])-0.1, max(X_std[:,0])+0.1]
[x1Min, x1Max] = [min(X_std[:,1])-0.1, max(X_std[:,1])+0.1]
delta = 0.01
[x0Plt, x1Plt] = np.meshgrid(np.arange(x0Min, x0Max, delta), np.arange(x1Min, x1Max, delta))
h = svm_clf.decision_function(np.c_[x0Plt.ravel(), x1Plt.ravel()])
h = h.reshape(x0Plt.shape)
CS = ax3.contour(x0Plt, x1Plt, h, cmap=plt.cm.twilight)
ax3.clabel(CS)












