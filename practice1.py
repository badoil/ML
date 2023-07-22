#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:17:21 2023

@author: oneal.oh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
import matplotlib.font_manager as mfm

D1 = np.array([[1.0, 1.2, 3, 4, 5, 6], [1.7, 3, 2.3, 5.3, 3.8, 5.5]])
D2 = np.array([[-0.6, 1.0, 1.2, 3, 4, 5, 6], [2.9, 1.7, 3, 2.3, 5.3, 3.8, 5.5]])

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches((15,6))

ax1.xaxis.set_tick_params(labelsize=18)
ax1.yaxis.set_tick_params(labelsize=18)
ax1.set_xlabel('', fontsize=25)
ax1.set_ylabel('', fontsize=25)
ax1.grid(False)

ax2.xaxis.set_tick_params(labelsize=18)
ax2.yaxis.set_tick_params(labelsize=18)
ax2.set_xlabel('', fontsize=25)
ax2.set_ylabel('', fontsize=25)
ax2.grid(False)

ax1.plot(D1[0], D1[1], 'ko', markersize=10)
ax1.set_xlim([-1,7])
ax1.set_ylim([1,6])
ax1.set_title('D1', fontsize=18)

ax2.plot(D2[0], D2[1], 'ko', markersize=10)
ax2.set_xlim([-1,7])
ax2.set_ylim([1,6])
ax2.set_title('D2', fontsize=18)

plt.show()


def machine_learning(D):
    """
    선형회귀 알고리즘을 사용하여 최적의 직선을 계산합니다.
    D : (2,N)의 어레이로 1행에는 데이터의 x좌표
        2행에는 데이터의 y좌표가 저장 되어 있습니다.
    """
    # 데이터의 개수를 N에 할당합니다.
    N = D.shape[1] 
    
    # 1열에 1, 2열에 데이터의 x좌표를 가지는 행렬을 만듭니다.
    # X: (N,2), y: (N,)
    X = np.c_[np.ones(N), D[0]]
    y = D[1]
    
    # 앞으로 배울 정규방정식을 풀어서 직선의 계수를 구합니다.
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return w

def more_clever(D):
    """
    첫점과 끝점을 연결하는 직선을 계산합니다.
    D : (2,N)의 어레이로 1행에는 데이터의 x좌표
        2행에는 데이터의 y좌표가 저장 되어 있습니다.
    """
    first, last = D[:,0], D[:,-1]
    
    w1 = (last[1]-first[1]) / (last[0]-first[0])
    w0 = -w1*first[0] + first[1]
    
    return (w0, w1)
    
def f(x, w):
    """
    주어진 w를 사용하여 직선 f(x) = w[1]*x + w[0]의 값을 계산합니다.
    """
    return w[1]*x + w[0]


# D1에 대해서 w[1]*x + w[0]에서 w[0], w[1]을 구합니다. 
w_ml_d1 = machine_learning(D1)
w_mc_d1 = more_clever(D1)

# D2에 대해서 w[1]*x + w[0]에서 w[0], w[1]을 구합니다. 
w_ml_d2 = machine_learning(D2)
w_mc_d2 = more_clever(D2)

print(w_ml_d1)

x = np.linspace(-1, 7, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches((15,6))

ax1.xaxis.set_tick_params(labelsize=18)
ax1.yaxis.set_tick_params(labelsize=18)
ax1.set_xlabel('', fontsize=25)
ax1.set_ylabel('', fontsize=25)
ax1.grid(False)

ax2.xaxis.set_tick_params(labelsize=18)
ax2.yaxis.set_tick_params(labelsize=18)
ax2.set_xlabel('', fontsize=25)
ax2.set_ylabel('', fontsize=25)
ax2.grid(False)

ax1.plot(D1[0], D1[1], 'ko', markersize=10)
ax1.plot(x, f(x, w_ml_d1), c='k', lw=2, label='machine learning')
ax1.plot(x, f(x, w_mc_d1), '--', c='k', lw=2, label='more clever')
ax1.set_xlim([-1,7])
ax1.set_ylim([1,6])
ax1.legend(fontsize=18)

ax2.plot(D2[0], D2[1], 'ko', markersize=10)
ax2.plot(x, f(x, w_ml_d2), c='k', lw=2, label='machine learning')
ax2.plot(x, f(x, w_mc_d2), '--', c='k', lw=2, label='more clever')
ax2.set_xlim([-1,7])
ax2.set_ylim([1,6])
ax2.legend(fontsize=18)

plt.show()




























