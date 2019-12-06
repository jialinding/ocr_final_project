from models import OCRModel
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
from tqdm import tqdm
from utils import Dataset, pack
from chars import *

train_dir = sys.argv[1]
batch_size = 64
num_epochs = 50
num_workers = 16

model = OCRModel(num_chars=NUM_TOKENS)
device = torch.device("cuda:2")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
encoder = model.encoder

train_dataset = Dataset(train_dir)
dataloader = torch.utils.data.DataLoader(train_dataset,
              batch_size=batch_size, shuffle=True, 
              collate_fn=pack,
              num_workers=num_workers) 
ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)
for _ in range(num_epochs):
  pbar = tqdm(iter(dataloader), total=len(dataloader))
  for x, y_padded, y_packed, y_lengths in pbar:
    x = x.to(device=device)
    y_padded = y_padded.to(device)
    y_packed = y_packed.to(device=device)

    logits = encoder(x, device)
    logits, input_lengths = nn.utils.rnn.pad_packed_sequence(
          logits,
          batch_first=False)
    # check the shape
    L, N, C = logits.shape
    assert (L, N, C) == (max(input_lengths), x.batch_sizes[0], NUM_TOKENS)

    N, L = y_padded.shape
    assert (N, L) == (x.batch_sizes[0], max(y_lengths))

    preds = logits.log_softmax(dim=2)
    loss = ctc_loss(preds, y_padded, input_lengths, torch.LongTensor(y_lengths).to(device))

    pbar.set_description('loss: %.4f' % float(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  torch.save(model.state_dict(), 'ocr.model')
