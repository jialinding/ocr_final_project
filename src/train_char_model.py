from models import OCRModel, initialize_char_model, CharModel
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
from tqdm import tqdm
from utils import Dataset, CharDataset, pack
from chars import *
import os
import numpy as np

def acc(outputs, labels):
  outputs = outputs.detach().numpy()
  labels = labels.numpy()
  outputs = outputs.argmax(axis=1)
  return (outputs == labels).astype(np.int).sum() / labels.shape[0]

train_dir, val_dir = sys.argv[1:]
batch_size = 10
num_epochs = 6
num_workers = 16

# model = initialize_char_model(num_chars=NUM_TOKENS)
model = CharModel(num_chars=NUM_TOKENS)
device = torch.device("cpu")
model.to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

train_dataset = CharDataset(train_dir)
val_dataset = CharDataset(val_dir)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
              batch_size=batch_size, shuffle=True,
              num_workers=num_workers) 
val_dataloader = torch.utils.data.DataLoader(val_dataset,
              batch_size=batch_size, shuffle=False,
              num_workers=num_workers) 
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
output_salt_number = 0
while os.path.exists('char.model.{}'.format(output_salt_number)):
  output_salt_number += 1
for epoch in range(num_epochs):
  running_loss = 0.0
  running_acc = 0.0
  for inputs, labels in tqdm(iter(train_dataloader), total=len(train_dataloader)):
    inputs = inputs.to(device=device)
    labels = labels.to(device=device)

    optimizer.zero_grad()

    outputs = model(inputs)
    labels = labels.reshape(-1)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    running_acc += acc(outputs, labels)
  
  print('[Epoch %d] train loss: %.3f acc: %.3f' % (epoch + 1, running_loss / len(train_dataloader), running_acc / len(train_dataloader)))
  torch.save(model.state_dict(), 'char.model.{}'.format(output_salt_number))

  running_loss = 0.0
  running_acc = 0.0
  for inputs, labels in tqdm(iter(val_dataloader), total=len(val_dataloader)):
    inputs = inputs.to(device=device)
    labels = labels.to(device=device)

    outputs = model(inputs)
    labels = labels.reshape(-1)
    loss = criterion(outputs, labels)

    running_loss += loss.item()
    running_acc += acc(outputs, labels)
  
  print('[Epoch %d] val loss: %.3f acc: %.3f' % (epoch + 1, running_loss / len(val_dataloader), running_acc / len(val_dataloader)))
