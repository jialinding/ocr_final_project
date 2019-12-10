import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class LineEncoder(nn.Module):
  '''
  [images of a line] -> <embedding for the line>
  '''
  def __init__(self, num_chars, hidden_size=64, num_channels=64):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(1, num_channels, 3),
        nn.ReLU(),
        nn.BatchNorm2d(num_channels),

        nn.Conv2d(num_channels, num_channels, 3),
        nn.ReLU(),
        nn.BatchNorm2d(num_channels),

        nn.Conv2d(num_channels, num_channels, 3),
        nn.ReLU(),
        nn.BatchNorm2d(num_channels),

        Flatten()
        )
    cnn_out_size = num_channels * 22 * 1
    self.lstm = nn.LSTM(
        input_size=cnn_out_size,
        hidden_size=hidden_size,
        batch_first=True, bidirectional=True)
    self.h_init = nn.Parameter(torch.randn(2, 1, hidden_size))
    self.c_init = nn.Parameter(torch.randn(2, 1, hidden_size))
    self.fc = nn.Sequential(
        nn.Linear(2 * hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_chars))

  def init(self, batch_size, device):
    h = self.h_init.repeat(1, batch_size, 1).to(device)
    c = self.c_init.repeat(1, batch_size, 1).to(device)
    return h, c

  def forward(self, x, device):
    # run the images through the CNN to extract features
    num_images = len(x.data)
    features = self.cnn(x.data.view(num_images, 1, 28, 7))
    assert len(features) == len(x.data)
    features = x._replace(data=features)

    # run the extracted features from the LSTM
    batch_size = x.batch_sizes[0]
    states, _ = self.lstm(features, self.init(batch_size, device))

    logits = self.fc(states.data)

    return states._replace(data=logits)

class LineDecoder(nn.Module):
  '''
  <line emb> -> [char]
  '''
  def __init__(self, hidden_size=64, num_chars=258):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_chars = num_chars
    self.lstm = nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        batch_first=True)
    self.char_emb = nn.Embedding(num_chars, hidden_size)
    self.fc = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_chars))

  def forward(self, x, hidden):
    embs = self.char_emb(x.data.view(-1))
    x = x._replace(data=embs)
    out, hidden = self.lstm(x, hidden)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    batch_size, max_len, _ = out.shape
    assert out.shape[2] == self.hidden_size
    out = out.contiguous().view(-1, self.hidden_size)
    logits = self.fc(out)
    y = logits.view(batch_size, max_len, self.num_chars)
    return y, hidden

class OCRModel(nn.Module):
  def __init__(self, num_chars, hidden_size=512, num_channels=128):
    super().__init__()
    self.encoder = LineEncoder(num_chars, 
        hidden_size=hidden_size, num_channels=num_channels)
    #self.decoder = LineDecoder(hidden_size=hidden_size, num_chars=num_chars)

# some tests to make sure the shapes match up
#from utils import pack_sequences
#x = torch.rand(1, 28, 7)
#seq1 = x.repeat(3,1,1)
#seq2 = x.repeat(4,1,1)
#seq3 = x.repeat(5,1,1)
#rcnn = LineEncoder(num_chars=101, hidden_size=64, num_channels=128)
#device = torch.device('cpu')
#rcnn(pack_sequences([seq1, seq2, seq3], device), device)
#decoder = LineDecoder(hidden_size=64)
#device = torch.device('cpu')
#lines = [torch.zeros(12).long(),
#    torch.zeros(13).long(),
#    torch.zeros(6).long()]
#decoder(pack_sequences(lines, device), hidden)

def initialize_char_model(num_chars):
  input_size = 28 * 28
  hidden_sizes = [128, 64]
  output_size = num_chars
  char_model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    # nn.LogSoftmax(dim=1)
  )
  return char_model

class CharModel(nn.Module):
  def __init__(self, num_chars):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 32, 3)
    self.fc1 = nn.Linear(32 * 5 * 5, 256)
    self.fc2 = nn.Linear(256, num_chars)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 32 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
