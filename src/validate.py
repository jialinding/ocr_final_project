import sys
from utils import Dataset, pack
from models import OCRModel
import torch
from tqdm import tqdm
from chars import *
import torch.nn as nn

from inference import decode

model_f, val_dir = sys.argv[1:]

# run the model in batches for high GPU utilization
batch_size = 128
# number of workers we use to load validation data
num_workers = 8

model = OCRModel(num_chars=NUM_TOKENS)
device = torch.device('cuda:0') if torch.has_cuda else torch.device('cpu')
model.load_state_dict(torch.load(model_f, map_location=device))
if torch.has_cuda:
  model.cuda()
model.to(device)
model.eval()

val_dataset = Dataset(val_dir)
dataloader = torch.utils.data.DataLoader(val_dataset,
              batch_size=batch_size,
              collate_fn=pack,
              num_workers=num_workers) 

pbar = tqdm(iter(dataloader), total=len(dataloader))
for x, y_padded, y_packed, y_lengths in pbar:
  x = x.to(device=device)
  y_padded = y_padded.to(device)
  y_packed = y_packed.to(device=device)

  logits = model.encoder(x, device)
  logits, input_lengths = nn.utils.rnn.pad_packed_sequence(
        logits,
        batch_first=True)
  N, L, C = logits.shape
  assert (N, L, C) == (x.batch_sizes[0], max(input_lengths), NUM_TOKENS)

  zipped = zip(logits.softmax(dim=2), input_lengths, y_padded, y_lengths)
  for preds, l, target, target_l in zipped:
    s_decoded = decode(preds[:l, :])
    s_target = ''.join(ids2chars[int(i)] for i in target[:target_l])
    print('DECODED:', s_decoded)
    print('LABEL:', s_target)
