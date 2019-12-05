from models import OCRModel, pack_sequences
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import glob
import cv2
import sys
from tqdm import tqdm
from utils import split_image
from chars import *

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dir):
    image_files = glob.glob(dir+'/*.jpg')
    images = []
    text_labels = {}

    for f in tqdm(image_files):
      label_name, qual, _ = f.split('.')
      label_name = label_name.split('/')[-1]
      images.append((f, label_name))
      if label_name not in text_labels:
        with open(dir + '/' + label_name + '.txt') as label_f:
          text_labels[label_name] = torch.LongTensor(
              [char2ids[c] for c in label_f.read().strip()])

    self.images = images
    self.labels = text_labels

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_f, label_name = self.images[idx]
    img = cv2.imread(img_f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = split_image(img)
    y = self.labels[label_name]
    return x, y

def pack(data):
  xs, ys = zip(*data)
  y_lengths = [len(y) for y in ys]
  y_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=BLANK)
  y_packed = nn.utils.rnn.pack_padded_sequence(
      y_padded, y_lengths,
      batch_first=True, enforce_sorted=False)
  return pack_sequences(xs, torch.device('cpu')), y_padded, y_packed, y_lengths

train_dir = sys.argv[1]
batch_size = 64
num_epochs = 50
num_workers = 16

model = OCRModel(num_chars=NUM_TOKENS)
device = torch.device("cuda:2")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
encoder = model.encoder
#decoder = model.decoder

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
