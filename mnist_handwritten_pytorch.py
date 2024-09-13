# -*- coding: utf-8 -*-
"""Mnist_Handwritten_PyTorch.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Y_qonVIPWf51-KnhVU9DsubKojKHIpuA
"""

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
type(x_train)

# Converting data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)

# Simple Linear Regression example
x = torch.tensor([1,2,3,4,5,6,7,8,9], dtype=torch.int64)
y = torch.tensor([2,4,6,8,10,12,14,16,18], dtype=torch.int64)

w = torch.tensor([0.0], requires_grad=True)

for epoch in range(100):
  y_pred = w * x
  loss = (y-y_pred)**2
  loss = loss.mean()
  loss.backward()
  with torch.no_grad():
    w -= 0.01 * w.grad
  w.grad.zero_()
  if epoch%10==0:
    print('Epoch:', epoch, ' Loss:', loss.item())

print(f'{(w*(20.0)).item()}')

"""# **Standard PyTorch Pipline**
1. Design a model with different layers
2. Construct Loss and Optimizer
3. Training Loop:
Forward pass - Prediction and Loss Calculation, Backward Pass - Calculate Gradients, Update Weights
"""

# MNIST dataset model

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Tensor Data to Dataset and DataLoader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class MultiClassClassifier(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.lin1 = nn.Linear(784, 392)
    self.lin2 = nn.Linear(392, 196)
    self.lin3 = nn.Linear(196, 98)
    self.relu = nn.ReLU()
    self.lin4 = nn.Linear(98, 10)

  def forward(self, x):
    l1 = self.lin1(x)
    l2 = self.lin2(l1)
    l3 = self.lin3(l2)
    l3 = self.relu(l3)
    l4 = self.lin4(l3)
    return l4

EPOCHS = 30
LEARNING_RATE = 0.001
model = MultiClassClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
  for x, y in train_dataloader:
    x = x.view(x.shape[0], -1).to(device)
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  if epoch%10==0:
      print('Epoch:', epoch, ' Loss:', loss.item())

with torch.no_grad():
  correct = 0
  for x, y in test_dataloader:
    x = x.view(-1, 784).to(device)
    y_pred = model(x)
    _, predicted = torch.max(y_pred.data, 1)
    correct += (predicted == y).sum().item()
  print('Accuracy:', correct/len(test_dataset))
