from models import OCRModel
import torch
import torch.nn as nn
import sys
import cv2
from utils import split_image, pack_sequences
from chars import *

MAX_LEN = 1000

def decode(preds):
  s = ''
  prev_c = None
  for prob in preds:
    _, indices = prob.topk(10)
    top_chars = [ids2chars[int(i)] for i in indices]
    c = top_chars[0]
    if prev_c == c:
      # merge consecutive characters
      continue

    if prev_c is not None:
      s += prev_c

    is_blank = indices[0] == BLANK
    if is_blank:
      prev_c = None
    else:
      prev_c = c

  return s

if __name__ == '__main__':
  model_f, img_f = sys.argv[1:]
  
  model = OCRModel(num_chars=NUM_TOKENS)
  model.load_state_dict(torch.load(model_f))
  model.eval()
  
  img = cv2.imread(img_f)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  device = torch.device('cpu')
  images = split_image(img)
  logits = model.encoder(pack_sequences([images], device), device)
  logits, input_lengths = nn.utils.rnn.pad_packed_sequence(
        logits,
        batch_first=False)

  s = decode(logits.view(len(images), -1))
  print(s)
