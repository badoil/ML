#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 13:51:42 2023

@author: oneal.oh
"""

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.optim as optim




# Load the DataFrame from a CSV file
df = pd.read_csv('./classify5k.csv')

# Convert the DataFrame to a numpy array
data = df[['x', 'y']].values
labels = df['label'].values.reshape(-1, 1)

# Print the shapes of the data and labels
print(f'Data shape:{data.shape}')
print(f'Labels shape:{labels.shape}')

# Plot the points
# plt.figure(figsize=(6, 6))
# plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis', s=1)
# plt.title('Sample Data with Noisy Labels')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

########################################################################

# Convert the numpy arrays to PyTorch tensors
data_torch = torch.tensor(data, dtype=torch.float32)
labels_torch = torch.tensor(labels, dtype=torch.float32)

print(f"before, one hot : {labels_torch.shape}")
labels_torch = F.one_hot(labels_torch)
print(f"after, one hot : {labels_torch.shape}")

data_length = len(data_torch)
split_length = int(0.8*data_length)

train_data = data_torch[:split_length]
train_labels = labels_torch[:split_length]
val_data = data_torch[split_length:]
val_labels = labels_torch[split_length:]

print(train_data.shape, train_labels.shape, train_labels.dtype)


# get batch
g = torch.Generator().manual_seed(42)

def get_batch(data, labels, batch_size=256):
    # Generate random indices
    indices = torch.randint(0, len(data), size=(batch_size,), generator= g)

    # Select the data and labels at these indices
    data_batch = data[indices]
    labels_batch = labels[indices]

    return data_batch, labels_batch

data_batch, labels_batch = get_batch(train_data,train_labels)
print(data_batch.shape, labels_batch.shape)



########################################################################


def custom_cross_entropy_loss(output, ont_hot_label):
    exp_output = torch.exp(output)
    total = torch.sum(exp_output, dim=1, keepdim=True)
    softmax_output = exp_output/total
    log_prob = - torch.log(softmax_output)
    nll = (log_prob * ont_hot_label).sum(dim=1)
    return torch.mean(nll)


########################################################################



# Define the size of each layer
input_size = 2  # The input size is 2 (x and y coordinates)
hidden_size = 4  # The size of the hidden layer
output_size = 1  # The output size is 1 (for binary classification)


# Define the network as a subclass of nn.Module
g.manual_seed(42)
torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq_model(x)

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)


# Net 클래스로 뉴럴네트워크를 만들기에 아래는 노필요
# W1 = torch.randn((input_size, hidden_size), generator=g)
# b1 = torch.randn(hidden_size, generator=g)
# W2 = torch.randn((hidden_size, output_size) , generator=g)
# b2 = torch.randn(output_size, generator=g)
# params = [W1,b1,W2,b2]
# for p in params:
#     p.requires_grad = True

for steps in range(200000):
    net.train() # train 모드로 바꿈
    data_batch, labels_batch = get_batch(train_data,train_labels, batch_size=256)
    output = net(data_batch)
    
    # forward propagation은 net 객체 통과시키는거로 아래를 대체
    # tmp = data_batch@W1 + b1
    # tmp = F.relu(tmp)
    # output = tmp@W2 + b2
    #prob = torch.sigmoid(output)
    
    loss = custom_cross_entropy_loss(output, labels_batch)
    # 로스함수를 대체
    # loss = -1 * (labels_batch * torch.log(prob) + (1 - labels_batch) * torch.log(1 - prob)).mean()


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 파라미터 업데이트를 optimizer 로 수행, 이미 optimizer에는 net의 파라미터 전달
    # loss.backward()
    # with torch.no_grad():
    #     W1 -= 0.01 * W1.grad
    #     b1 -= 0.01 * b1.grad
    #     W2 -= 0.01 * W2.grad
    #     b2 -= 0.01 * b2.grad
    #     W1.grad.zero_()
    #     b1.grad.zero_()
    #     W2.grad.zero_()
    #     b2.grad.zero_()

    if steps % 1000 == 0:
        net.eval()
        output = net(val_data)
        val_loss = custom_cross_entropy_loss(output, val_labels)
        print(f"{steps} val_loss: {val_loss.item()}")

        # tmp = val_data@W1 + b1
        # tmp = F.relu(tmp)
        # output = tmp@W2 + b2
        # prob = torch.sigmoid(output)
        # val_loss = -1 * (val_labels * torch.log(prob) + (1 - val_labels) * torch.log(1 - prob)).mean()
        #  print(f"{steps},val_loss: {val_loss},train_loss: {loss}" )



















