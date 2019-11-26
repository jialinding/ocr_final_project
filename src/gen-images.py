from utils import gen_image
import random
import sys
import cv2
import os
from tqdm import tqdm

counter = 0

qualities = list(range(1, 20, 2))

def downsample(img):
  h, w, _ = img.shape
  return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

# TODO: perturb the generated image
def output_chunk(source_name, lines, output_dir, post_processing=downsample):
  global counter
  name = '%s_%d' % (source_name, counter)
  counter += 1

  img = gen_image(lines)
  img = post_processing(img)

  # write the generated image with various quality
  for qual in qualities:
    out_name = '%s.%d.jpg' % (name, qual)
    cv2.imwrite(os.path.join(output_dir, out_name), img, (cv2.IMWRITE_JPEG_QUALITY, qual))

  # write the text, which will be the labels.
  with open(os.path.join(output_dir, name+'.txt'), 'w') as text_f:
    for line in lines:
      text_f.write(line+'\n')

input_f, output_dir = sys.argv[1:]

min_chunk_size = 25
max_chunk_size = 100

chunk = []
chunk_size = random.randint(min_chunk_size, max_chunk_size)
source_name = os.path.basename(input_f).split('.')[0]
with open(input_f) as f:
  for line in tqdm(list(f)):
    chunk.append(line[:-1]) # strip '\n'
    if len(chunk) == chunk_size:
      output_chunk(source_name, chunk, output_dir)
      chunk_size = random.randint(min_chunk_size, max_chunk_size)
      chunk = []
