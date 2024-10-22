# OM NAMAH SHIVAY
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
from torch.distributed import init_process_group, destroy_process_group, barrier
import os
import socket
import argparse


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


def ddp_setup(g_rank, world_size):
  print("Inside ddp setup")
  # ip_address = get_ip_address()
  # available_port = find_free_port()
  os.environ['MASTER_ADDR'] = '10.10.1.1'
  os.environ['MASTER_PORT'] = '29500'
  # print("IP Addr:", os.environ['MASTER_ADDR'], " Port:", os.environ['MASTER_PORT'])
  init_process_group(backend='nccl', rank=g_rank, world_size=world_size)


# In[5]:


def prepare_data(world_size, g_rank, l_rank, batch_size=32):

  download = True if l_rank == 0 else False
  transform = transforms.Compose([transforms.ToTensor()])
  if l_rank == 0:
      train_dataset = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
      test_dataset = datasets.MNIST(root='./data', train=False, download=download, transform=transform)
  barrier()
  if l_rank != 0:
      train_dataset = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
      test_dataset = datasets.MNIST(root='./data', train=False, download=download, transform=transform)
      
  train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=g_rank)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, sampler=train_sampler)


  test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=g_rank)
  test_dataloader = DataLoader(test_dataset, batch_size=32, drop_last=True, sampler=test_sampler)
  return train_dataloader, test_dataloader



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
    l1 = self.relu(l1)
    l2 = self.lin2(l1)
    l2 = self.relu(l2)
    l3 = self.lin3(l2)
    l3 = self.relu(l3)
    l4 = self.lin4(l3)
    return l4


# In[8]:


def train_model(model, train_dataloader, optimizer, loss_fn, device):
  for epoch in range(3):
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


def evaluate(model, test_dataloader, device):
  with torch.no_grad():
    correct = 0
    for x, y in test_dataloader:
      x = x.view(-1, 784).to(device)
      y_pred = model(x)
      _, predicted = torch.max(y_pred.data, 1)
      correct += (predicted == y.to(device)).sum().item()
    print('Test Data Accuracy:', correct/len(test_dataloader))


# In[21]:


def main(l_rank, world_size, node_rank, n_cores):

  g_rank = l_rank + (node_rank * n_cores)
  
  print("Inside main")
  ddp_setup(g_rank, world_size)
  print("DDP Setup complete")

  device = f'cuda:{l_rank}'
  train_dataloader, test_dataloader = prepare_data(world_size=world_size, g_rank=g_rank, l_rank=l_rank, batch_size=32)
  model = MultiClassClassifier().to(device)
  model = DDP(model, device_ids=[l_rank])
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  loss_fn = nn.CrossEntropyLoss()
  model = train_model(model, train_dataloader, optimizer, loss_fn, device)
  
  barrier()
    
  evaluate(model, test_dataloader, device)

  destroy_process_group()

  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Elapsed time: {elapsed_time} seconds")


# In[22]:


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument('--node_id', type=int, required=True)
  parser.add_argument('--n_nodes', type=int, default=2)
  parser.add_argument('--n_cores', type=int, default=2)
  parser.add_argument('--node_rank', type=int, required=True)
  parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
  args = parser.parse_args()
  world_size = args.n_nodes * args.n_cores
  mp.spawn(main, args=(world_size, args.node_rank, args.n_cores), nprocs=args.n_cores)


# In[ ]:
