import cv2
import numpy as np
import random
import torch

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
  # split the image into a list of 28 x 14 smaller images
  orig = img
  img = torch.from_numpy(img).float() / 255
  h, w = img.shape
  assert h == 28
  images = []
  for i in range(0, w, 14):
    if i + 14 <= w:
      images.append(img[:, i:i+14])
    else:
      padded = torch.ones(28,14)
      padded[:, :w-i] = img[:, i:w]
      images.append(padded)
  return torch.stack(images)
