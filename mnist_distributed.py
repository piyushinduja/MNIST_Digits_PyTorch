#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.datasets
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
start_time = time.time()

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import socket


# In[4]:


def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# In[12]:


def ddp_setup(rank, world_size):
  print("Inside ddp setup")
  ip_address = get_ip_address()
  available_port = find_free_port()
  os.environ['MASTER_ADDR'] = ip_address
  os.environ['MASTER_PORT'] = '8888'
  print("IP Addr:", os.environ['MASTER_ADDR'], " Port:", os.environ['MASTER_PORT'])
  init_process_group(backend='gloo', rank=rank, world_size=world_size)


# In[5]:


def prepare_data(world_size, rank, batch_size=32, pin_memory=False, num_workers=0):
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
  train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=train_sampler)

  test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
  test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, sampler=test_sampler)
  return train_dataloader, test_dataloader


# # **Standard PyTorch Pipline**
# 1. Design a model with different layers
# 2. Construct Loss and Optimizer
# 3. Training Loop:
# Forward pass - Prediction and Loss Calculation, Backward Pass - Calculate Gradients, Update Weights

# In[7]:


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


# In[8]:


def train_model(model, train_dataloader, optimizer, loss_fn, rank, device):
  for epoch in range(5):
    train_dataloader.sampler.set_epoch(epoch)
    for x, y in train_dataloader:
      x = x.view(x.shape[0], -1).to(device)
      y_pred = model(x)
      loss = loss_fn(y_pred, y.to(device))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    print('Epoch:', epoch, ' Loss:', loss.item())
  return model


# In[9]:


def evaluate(model, test_dataloader, rank, device):
  with torch.no_grad():
    correct = 0
    for x, y in test_dataloader:
      x = x.view(-1, 784).to(device)
      y_pred = model(x)
      _, predicted = torch.max(y_pred.data, 1)
      correct += (predicted == y.to(device)).sum().item()
    print('Test Data Accuracy:', correct/len(test_dataloader))


# In[21]:


def main(rank, world_size):
  print("Inside main")
  ddp_setup(rank, world_size)
  print("DDP Setup complete")

  device = f'cpu:{rank}'
  train_dataloader, test_dataloader = prepare_data(world_size=world_size, rank=rank, batch_size=32, pin_memory=False, num_workers=0)
  model = MultiClassClassifier().to(device)
  model = DDP(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  loss_fn = nn.CrossEntropyLoss()
  model = train_model(model, train_dataloader, optimizer, loss_fn, rank, device)
  evaluate(model, test_dataloader, rank, device)

  destroy_process_group()

  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Elapsed time: {elapsed_time} seconds")


# In[22]:


if __name__ == '__main__':
  world_size = min(os.cpu_count(), 2)
  print(os.cpu_count())
  print("World Size:", world_size)
  # mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
  mp.spawn(main, args=(world_size,), nprocs=world_size)


# In[ ]:
