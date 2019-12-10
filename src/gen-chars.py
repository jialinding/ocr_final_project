from utils import gen_image, gen_image_all
from segment import extract_lines, extract_chars
import random
import sys
import cv2
import os
from tqdm import tqdm
from chars import char2ids, ids2chars
import numpy as np

counter = 0

qualities = list(range(1, 40, 2))

def downsample(img):
  h, w = img.shape
  # w = int(w / h * 28)
  w = 28
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
  detected_chars,_ = extract_chars(img)
  if len(detected_chars) == 0:
    return
  img = detected_chars[0]
  img = post_processing(img)

  # write the generated image with various quality
  for qual in qualities:
    out_name = '%s.%d.jpg' % (name, qual)
    cv2.imwrite(os.path.join(output_dir, out_name), img, (cv2.IMWRITE_JPEG_QUALITY, qual))

  # write the text, which will be the labels.
  with open(os.path.join(output_dir, name+'.txt'), 'w') as text_f:
    text_f.write(c+'\n')

def output_char_all(source_name, c, output_dir, post_processing=downsample, noise_samples=0, sp_samples=0):
  global counter

  imgs = gen_image_all([c])
  for img in imgs:
    name = '%s_%d' % (source_name, counter)
    counter += 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_lines = extract_lines(img)
    if len(detected_lines) == 0:
      continue
    img = detected_lines[0]
    detected_chars,_ = extract_chars(img)
    if len(detected_chars) == 0:
      continue
    img = detected_chars[0]
    img = post_processing(img)

    # write the generated image with various quality
    for qual in qualities:
      out_name = '%s.%d.jpg' % (name, qual)
      cv2.imwrite(os.path.join(output_dir, out_name), img, (cv2.IMWRITE_JPEG_QUALITY, qual))

    for i in range(noise_samples):
      noisy_img = img + np.random.normal(0, 25, (28, 28))
      for qual in qualities[::2]:
        out_name = '%s.%d-n%d.jpg' % (name, qual, i)
        cv2.imwrite(os.path.join(output_dir, out_name), noisy_img, (cv2.IMWRITE_JPEG_QUALITY, qual))

    for i in range(sp_samples):
      sp_img = img[:,:]
      num_salt = np.ceil(28 * 28 * 0.1)
      coords = [np.random.randint(0, s - 1, int(num_salt)) for s in img.shape]
      sp_img[coords] = 255
      num_pepper = np.ceil(28 * 28 * 0.1)
      coords = [np.random.randint(0, s - 1, int(num_salt)) for s in img.shape]
      sp_img[coords] = 0
      for qual in qualities[::2]:
        out_name = '%s.%d-sp%d.jpg' % (name, qual, i)
        cv2.imwrite(os.path.join(output_dir, out_name), sp_img, (cv2.IMWRITE_JPEG_QUALITY, qual))

    # write the text, which will be the labels.
    with open(os.path.join(output_dir, name+'.txt'), 'w') as text_f:
      text_f.write(c+'\n')

output_dir = sys.argv[1]

# for c, i in char2ids.items():
#   output_char_all('char_%d' % i, c, output_dir, noise_samples=2, sp_samples=2)

num_samples = 4
for c, i in char2ids.items():
  for _ in range(num_samples):
    output_char('char_%d' % i, c, output_dir)
