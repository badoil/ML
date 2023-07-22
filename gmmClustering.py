#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 14:16:07 2023

@author: oneal.oh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import scipy.stats


plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/ClassificationSample2.txt", 
                      sep="\s+")

samples = np.array(dfLoad)
x = samples[:, 0]
y = samples[:, 1]

N = len(x)
numk = 2
pi = np.ones(numk)*(1/numk)
mx = np.mean(x)
sx = np.std(x)
my = np.mean(y)
sy = np.std(y)

u0 = np.array([mx+sx, my+sy])
u1 = np.array([mx-sx, my-sy])
sigma0 = np.array([[sx*sx/4, 0], [0, sy*sy/4]])
sigma1 = np.array([[sx*sx/4, 0], [0, sy*sy/4]])

f1 = plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.plot(x, y, "b.")
ax1.plot([u0[0], u1[0]], [u0[1], u1[1]], "r*")


R = np.ones([N, numk]) * (1/numk)   # k로 분류될 확률, 처음에는 모든 값을 동일한 확률로 초기화
j=0
while(True):
    j+=1
    N0 = sp.stats.multivariate_normal.pdf(samples, u0, sigma0)  # 평균과 표준편차를 따르는 다차원 가우시안 확률함수 초기화
    N1 = sp.stats.multivariate_normal.pdf(samples, u1, sigma1)  # 여기서 평균과 표준편차를 업데이트하면서 밑에서 R값을 최신화
    
    # E-step, expectation step
    Rold = np.copy(R)
    R = np.array([pi[0]*N0/(pi[0]*N0 + pi[1]*N1), pi[1]*N1/(pi[0]*N0 + pi[1]*N1)]).T
    
    if (np.linalg.norm(R-Rold) < N * numk * 0.0001):
        break
    
    r0 = sum(R[:, 0])
    r1 = sum(R[:, 1])
    
    # M-step, maximization step
    pi[0] = r0/N    # 클러스터링하는 파이값을 업데이트
    pi[1] = r1/N
    # pi = np.ones(N).reshape(1,N).dot(R)/N
    # pi = pi.reshape(2,)
    weightedSum = samples.T.dot(R)
    u0 = weightedSum[:, 0]/r0      # 평균 업데이트
    u1 = weightedSum[:, 1]/r1
    sigma0 = samples.T.dot(np.multiply(R[:, 0].reshape(N,1), samples)) / r0 - u0.reshape(2,1)*u0.reshape(2,1).T
    sigma1 = samples.T.dot(np.multiply(R[:, 1].reshape(N,1), samples)) / r1 - u1.reshape(2,1)*u1.reshape(2,1).T

    
# clustering 한 결과 visualize
k = np.round(R[:, 1])
dfCluster = pd.DataFrame(np.c_[x, y, k])
dfCluster.columns=["X", "Y", "K"]
dfGroup = dfCluster.groupby("K")

f2 = plt.figure(2)
ax2 = f2.add_subplot(111)
for (cluster, dataGroup) in dfGroup :
    ax2.plot(dataGroup.X, dataGroup.Y, ".", label=cluster)












