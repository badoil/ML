#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:01:24 2023

@author: oneal.oh
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/ClassificationSample.txt", 
                      sep="\s+")

samples = np.array(dfLoad)
x = np.array(dfLoad["X"])
y = np.array(dfLoad["Y"])

f1 = plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.plot(x, y, "b. ")


N = len(x)

[mx, sx] = [np.mean(x), np.std(x)]
[my, sy] = [np.mean(y), np.std(y)]
z0 = np.array([mx+sx, my+sy]).reshape(1,2)
z1 = np.array([mx-sx, my-sy]).reshape(1,2)
Z = np.r_[z0, z1]
ax1.plot(Z[:, 0], Z[:, 1], "r* ", markersize = "20")


k = np.zeros(N)
while(1):
    kOld = np.copy(k)
    for i in np.arange(N):
        z0d = np.linalg.norm(samples[i, :] - Z[0, :])
        z1d = np.linalg.norm(samples[i, :] - Z[1, :])
        k[i] = z0d > z1d
        
    if (np.alltrue(k == kOld)):
        break
        
    dfCluster = pd.DataFrame(np.c_[x, y, k])
    dfCluster.columns = ["X", "Y", "K"]
    dfGroup = dfCluster.groupby("K")    

    for cluster in range(2):
        Z[cluster, :] = dfGroup.mean().iloc[cluster]

ax1.plot(Z[:, 0], Z[:, 1], "g* ", markersize = "20")


f2 = plt.figure(2)
ax2 = f2.add_subplot(1,1,1)
ax2.plot(Z[:, 0], Z[:, 1], "r*", markersize="20")
for clusterName, group in dfGroup:
    ax2.plot(group.X, group.Y, ".", label=clusterName)
    
ax2.legend()

# for group, dataInCluster in dfGroup:
#     print(group, dataInCluster)


# 클러스터 연

# N = len(x)
# np.random.seed(3)
# k = np.round(np.random.rand(N))

# npCluster = np.c_[x, y, k]
# dfCluster = pd.DataFrame(npCluster)
# dfCluster.columns = ["X", "Y", "K"]

# dfGroup = dfCluster.groupby(k)
# for group, dataInCluster in dfGroup:
#     print(group, dataInCluster)