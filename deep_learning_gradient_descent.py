#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 22:28:49 2023

@author: oneal.oh
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pandas as pd


# def grad_descent(f, df, init_x, learning_rate=0.01, step=100):
#     x = init_x
#     x_log, y_log = [x], [f(x)]
    
#     for i in range(step):
#         grad = df(x)
#         x -= learning_rate*grad
        
#         x_log.append(x)
#         y_log.append(f(x))
    
#     return x_log, y_log


# def f2(x):
#     return 0.01*x**4 - 0.3*x**3 - 1.0*x + 10.0

# def df2(x):
#     return 0.04*x**3 - 0.9*x**2 - 1.0



# x_init = 2
# x_log, y_log = grad_descent(f2, df2, x_init)

# plt.scatter(x_log, y_log, color='red')
# x = np.arange(-5, 30, 0.01)
# plt.plot(x, f2(x))
# plt.xlim(-5, 30)
# plt.grid()
# plt.show()


# plt.style.use('seaborn-whitegrid')

# epoch = 1000
# lr = 0.1

# def sigmoid(x):
#     return 1/(1+np.exp(-x))


# def mean_squared_error(pred_y, true_y):
#     return 0.5 * (np.sum((pred_y - true_y)**2))


# def cross_entropy_error(pred_y, true_y):
#     if true_y.ndim == 1:
#         true_y = true_y.reshape(1, -1)
#         pred_y = pred_y.reshape(1, -1)
        
#     delta = 1e-7
#     return -np.sum(true_y * np.log(pred_y+delta))


# def cross_entropy_error_for_batch(pred_y, true_y):
#     if true_y.ndim == 1:
#         true_y = true_y.reshape(1, -1)
#         pred_y = pred_y.reshape(1, -1)
        
#     delta = 1e-7
#     batch_size = pred_y.shape[0]
#     return -np.sum(true_y * np.log(pred_y+delta)) / batch_size

# def cross_entropy_error_for_binary(pred_y, true_y):
#     return 0.5 * np.sum((-true_y * np.log(pred_y) - (1 - true_y) * np.log(1-pred_y)))

# def soft_max(x):
#     exp_x = np.exp(x)
#     exp_x_sum = np.sum(exp_x)
#     y = exp_x / exp_x_sum
#     return y


# def differential (f, x):
#     eps = 1e-5
#     diff_value = np.zeros_like(x)
    
#     for i in range(x.shape[0]):
#         temp = x[i]
        
#         x[i] = temp + eps
#         f1 = f(x)
        
#         x[i] = temp - eps
#         f2 = f(x)
        
#         diff_value[i] = (f1 - f2) / (2*eps) 
#         x[i] = temp

#     return diff_value


# class LogicGateNet():
#     def __init__(self):
#         def weight_init():
#             np.random.seed(1)
#             weight = np.random.randn(2)
#             bias = np.random.rand(1)
#             return weight, bias
        
#         self.weight, self.bias = weight_init()
    
#     def predict(self, x):
#         W = self.weight.reshape(-1, 1)
#         b = self.bias
#         y = sigmoid(np.dot(x, W) + b)
#         return y
    
#     def loss(self, x, true_y):
#         pred_y = self.predict(x)
#         return cross_entropy_error_for_binary(pred_y, true_y)
    
#     def get_gradient(self, x, t):
#         def loss_grad(grad):
#             return self.loss(x, t)
#         grad_W = differential(loss_grad, self.weight)
#         grad_B = differential(loss_grad, self.bias)
#         return grad_W, grad_B
        
        

# AND = LogicGateNet()

# X = np.array([[0,0], [0, 1], [1,0], [1,1]])
# Y = np.array([[0], [0], [0], [1]])

# train_loss_list = list()

# for i in range(epoch):
#     grad_W, grad_B = AND.get_gradient(X, Y)
    
#     AND.weight -= lr * grad_W
#     AND.bias -= lr *grad_B

#     loss = AND.loss(X, Y)
#     train_loss_list.append(loss)

#     if i % 100 == 99:
#         print("count{}, cost{}, weight{}, bias{}".format(i, loss, AND.weight, AND.bias))



(x_train, y_train), (x_test, y_test)= mnist.load_data()

def flatten_for_mnist(x):
    temp = np.zeros((x.shape[0], x[0].size))

    for idx, data in enumerate(x):
        temp[idx, :] = data.flatten()
        
    return temp

x_train, x_test = x_train/255.0, x_test/255.0

x_train = flatten_for_mnist(x_train)
x_test = flatten_for_mnist(x_test)

y_train = tf.one_hot(y_train, depth=10).numpy()
y_test = tf.one_hot(y_test, depth=10).numpy()





































