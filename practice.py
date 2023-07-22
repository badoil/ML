#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 22:06:21 2023

@author: oneal.oh
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as mfm
from matplotlib.patches import Rectangle, Circle, PathPatch, Arc, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


def f1(X):
    return (X[0]+2*X[1]-7)**2 + (2*X[0]+X[1]-5)**2 

def df1(X):
    dx0 = 2*(X[0]+2*X[1]-7)+4*(2*X[0]+X[1]-5)
    dx1 = 4*(X[0]+2*X[1]-7)+2*(2*X[0]+X[1]-5)
    return np.array([dx0, dx1])
    
def f2(X):
    """
    x    : (2,)인 2차원 ndarray 변수
    -----------------------------------------
    반환 : 로젠브록 함수값
    """
    return 50*(X[1]-X[0]**2)**2 + (2-X[0])**2

def df2(X):
    """
    x    : (2,)인 2차원 ndarray 변수
    -----------------------------------------
    반환 : 로젠브록 함수의 도함수 값
    """
    dx0 = -200*X[0]*(X[1]-X[0]**2)-2*(2-X[0])
    dx1 = 100*(X[1]-X[0]**2)
    return np.array([dx0, dx1])


from scipy.optimize import line_search

# 1. 초기화: 시작점 x^(0)를 선정
x = np.array([0, 4.5])

# 수렴 상수 ϵ 설정
def SDM(f, df, x, eps=1.0e-7, callback=None):
    max_iter = 10000
    
    # 반복 횟수 k=0으로 설정
    for k in range(max_iter):
        # 2. 경사도벡터 계산: c^(k) = ∇f(x^(k))를 계산
        c = df(x)

        # 3. 수렴판정: c^(k)<ϵ이면 x^*=x^(k)로 두고 정지, 아니면 단계를 계속 진행
        if np.linalg.norm(c) < eps :
            print("Stop criterion break Iter.: {:5d}, x: {}".format(k, x))
            break

        # 3. 강하방향 설정: d^(k)=-c^(k)
        d = -c 

        # 4. 이동거리 계산: d^(k)를 따라 f(α)=f(x^(k)+α*d^(k))를 최소화하는 α_k를 계산
        alpha = line_search(f, df, x, d)[0]
        # alpha = golden(f_alpha, args=(f, x, d))
            
        # 5. 업데이트: x^(k+1)=(x^(k)+α_k*d^(k)로 변수를 업데이트하고 
        #              k=k+1로 두고 2로 가서 반복
        x = x + alpha * d
        
        # 외부함수를 실행 알고리즘과는 상관없고 사용자에 따라
        # 실행하고 싶은 작업이 있으면 callback함수로 실행시키기 위함
        if callback :
            callback(x)    
    else:
        print("Stop max iter:{:5d} x:{}".format(k, x))

SDM(f1, df1, x)  
