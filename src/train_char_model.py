from models import OCRModel, initialize_char_model
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
from tqdm import tqdm
from utils import Dataset, CharDataset, pack
from chars import *

train_dir = sys.argv[1]
batch_size = 64
num_epochs = 10
num_workers = 16

model = initialize_char_model(num_chars=NUM_TOKENS)
device = torch.device("cuda:2")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
encoder = model.encoder

train_dataset = CharDataset(train_dir)
dataloader = torch.utils.data.DataLoader(train_dataset,
              batch_size=batch_size, shuffle=True,
              num_workers=num_workers) 
criterion = nn.CrossEntropyLoss()
for _ in range(num_epochs):
  running_loss = 0.0
  for inputs, labels in tqdm(iter(dataloader), total=len(dataloader)):
    inputs = inputs.to(device=device)
    labels = labels.to(device=device)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  
  print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss))

  torch.save(model.state_dict(), 'char.model')
