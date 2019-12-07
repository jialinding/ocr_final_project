import cv2
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import glob
from chars import *

font_face_pool = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ]
font_scale_pool = list(range(1, 5))
thickness_pool = list(range(2, 3))
color_pool = [(0,255)]

def gen_image(lines):
  font_face = random.choice(font_face_pool)
  font_scale = random.choice(font_scale_pool)
  thickness = random.choice(thickness_pool)
  text_c, background_c = random.choice(color_pool)

  # figure out size of the image
  ws = []
  hs = []
  for line in lines:
    (w, h), _ = cv2.getTextSize(line, font_face, font_scale, thickness)
    ws.append(w)
    hs.append(h)
  w = max(ws)
  h = sum(hs)

  img = np.zeros((int(h*2),int(w*1.5),3), dtype=np.uint8)
  img[:,:,:] = background_c

  y = 0
  for line, h2 in zip(lines, hs):
    y += int(h2 * 1.5)
    cv2.putText(img, line, (int(0.25*w), y), font_face, font_scale, (text_c,text_c,text_c), thickness)
  return img

def split_image(img):
  # split the image into a list of 28 x 7 smaller images
  orig = img
  img = torch.from_numpy(img).float() / 255
  h, w = img.shape
  assert h == 28
  images = []
  for i in range(0, w, 7):
    if i + 7 <= w:
      images.append(img[:, i:i+7])
    else:
      padded = torch.ones(28,7)
      padded[:, :w-i] = img[:, i:w]
      images.append(padded)
  return torch.stack(images)

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

def pack_sequences(seqs, device):
  lengths = [len(s) for s in seqs]
  seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
  return nn.utils.rnn.pack_padded_sequence(
      seqs_padded, lengths,
      batch_first=True, enforce_sorted=False)

def pack(data):
  '''
  pack dataset
  '''
  xs, ys = zip(*data)
  y_lengths = [len(y) for y in ys]
  y_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=BLANK)
  y_packed = nn.utils.rnn.pack_padded_sequence(
      y_padded, y_lengths,
      batch_first=True, enforce_sorted=False)
  return pack_sequences(xs, torch.device('cpu')), y_padded, y_packed, y_lengths

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=5):
  lr = init_lr * (0.1**(epoch // lr_decay_epoch))

  if epoch % lr_decay_epoch == 0:
      print('LR is set to {}'.format(lr))

  for param_group in optimizer.param_groups:
      param_group['lr'] = lr

  return optimizer
