from models import OCRModel, pack_sequences
import torch
import torch.nn as nn
import sys
import cv2
from utils import split_image
from chars import *

MAX_LEN = 1000

def transcribe(model, img):
  device = torch.device('cpu')
  images = split_image(img)
  logits = model.encoder(pack_sequences([images], device), device)
  logits, input_lengths = nn.utils.rnn.pad_packed_sequence(
        logits,
        batch_first=False)
  s = ''
  for prob in logits.view(len(images), -1):
    _, indices = prob.topk(10)
    print([ids2chars[int(i)] for i in indices])
  print(s)

model_f, img_f = sys.argv[1:]

model = OCRModel(num_chars=NUM_TOKENS)
model.load_state_dict(torch.load(model_f, map_location=torch.device('cpu')))
model.eval()

img = cv2.imread(img_f)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
s = transcribe(model, img)
print(s)
