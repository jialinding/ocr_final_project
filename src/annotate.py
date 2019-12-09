from models import OCRModel
import numpy as np
import torch
import torch.nn as nn
import sys
import cv2
from utils import split_image, pack_sequences
from chars import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
  model_f, img_f = sys.argv[1:]
  
  model = OCRModel(num_chars=NUM_TOKENS)
  device = torch.device('cpu')
  model.load_state_dict(torch.load(model_f, map_location=device))
  model.eval()
  
  img = cv2.imread(img_f)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  images = split_image(img)
  logits = model.encoder(pack_sequences([images], device), device)
  logits, input_lengths = nn.utils.rnn.pad_packed_sequence(
        logits,
        batch_first=False)

  probs, ids = logits.view(len(images), -1).softmax(dim=1).max(dim=1)
  chars = [ids2chars[int(i)] for i in ids]

  h, w = img.shape
  ratio = 2
  img = cv2.resize(img, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
  plt.imshow(img, cmap='gray')

  x = [14*i+7/2 for i in range(len(images))]
  heights = probs.detach().numpy()
  plt.bar(x, heights, tick_label=chars)

  plt.xticks(fontsize=20)
  plt.show()
