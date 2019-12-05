from utils import gen_image
from segment import extract_lines
import random
import sys
import cv2
import os
from tqdm import tqdm
from chars import char2ids, ids2chars

counter = 0

qualities = list(range(1, 40, 2))

def downsample(img):
  h, w = img.shape
  w = int(w / h * 28)
  h = 28
  return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

# TODO: perturb the generated image
def output_char(source_name, c, output_dir, post_processing=downsample):
  global counter
  name = '%s_%d' % (source_name, counter)
  counter += 1

  img = gen_image([c])
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  detected_lines = extract_lines(img)
  if len(detected_lines) == 0:
    return
  img = detected_lines[0]
  img = post_processing(img)

  # write the generated image with various quality
  for qual in qualities:
    out_name = '%s.%d.jpg' % (name, qual)
    cv2.imwrite(os.path.join(output_dir, out_name), img, (cv2.IMWRITE_JPEG_QUALITY, qual))

  # write the text, which will be the labels.
  with open(os.path.join(output_dir, name+'.txt'), 'w') as text_f:
    text_f.write(c+'\n')

output_dir = sys.argv[1]

num_samples = 16

for c, i in char2ids.items():
  for _ in range(num_samples):
    output_char('char_%d' % i, c, output_dir)
