import numpy as np
import cv2
import sys
import os

# Given two interval a and b, return length of the intersection
def intersect_interval(a, b):
  a_begin, a_end = a
  b_begin, b_end = b
  if a_end < b_begin or b_end < a_begin:
    return 0
  return min(a_end, b_end) - max(a_begin, b_begin)

def overlaps(box1, box2, threshold=0.8):
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2
  height_union = max(y1+h1, y2+h2) - min(y1,y2)
  height_intersection = intersect_interval((y1, y1+h1), (y2,y2+h2))
  overlap_ratio = height_intersection / min(height_union, h1, h2)
  return overlap_ratio > threshold

def union(box1, box2):
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2

  x = min(x1, x2)
  y = min(y1, y2)
  w = max(x1+w1, x2+w2) - x
  h = max(y1+h1, y2+h2) - y
  return x, y, w, h

# Takes img, returns list of imgs (each representing a row of text)
# Assumes img is grayscale
def extract_lines(img, draw=False):
  # binary
  ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

  # dilation
  kernel = np.ones((5,100), np.uint8)
  img_dilation = cv2.dilate(thresh, kernel, iterations=1)

  # find contours
  res = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if len(res) == 2:
    ctrs, hier = res
  else:
    _, ctrs, hier = res

  # sort by height
  sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

  prev_box = None

  lines = []

  for i, ctr in enumerate(sorted_ctrs):
      # Get bounding box
      x, y, w, h = cv2.boundingRect(ctr)
      cur_box = x, y, w, h

      if prev_box is None:
        prev_box = cur_box
        continue

      # merge with the previous box if overlaps significantly
      if overlaps(prev_box, cur_box):
        prev_box = union(prev_box, cur_box)
        cur_box = None
        continue

      # output the box
      x, y, w, h = prev_box
      lines.append(img[y:y+h, x:x+w])
      if draw:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
      prev_box = None
      prev_box = cur_box

  if prev_box is not None:
    x, y, w, h = prev_box
    lines.append(img[y:y+h, x:x+w])
    if draw:
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    prev_box = None

  if draw:
    return lines, img
  return lines

# Takes img representing row of text, returns list of imgs (each reepreseting a character)
# Assumes img is grayscale
# Spaces are represented as "None" in the returned list
# Assumes that characters are separated by white space,
# and each character is one connected piece of non-white space
# Based on https://stackoverflow.com/questions/10964226/how-to-convert-an-img-into-character-segments
def extract_chars(img, draw=False):
  # smooth the img to avoid noises
  # img = cv2.medianBlur(img,5)

  # Apply adaptive threshold
  thresh = cv2.adaptiveThreshold(img,255,1,1,11,2)

  # Find the contours
  res = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  if len(res) == 2:
    contours,hierarchy = res
  else:
    _, contours,hierarchy = res

  if len(contours) == 0:
    return []

  # Extract the x-ranges of each contour
  noise_threshold = 4
  x_ranges = []
  while len(x_ranges) == 0:
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # Ignore noisy results
        if w < noise_threshold or h < noise_threshold:
          continue

        x_ranges.append((x, x+w, y, y+h))
    noise_threshold -= 1

  # Sort x ranges by starting x value, and merge contours that overlap significantly.
  # This handles cases like "i" or "j" which are composed of two contours because of the dot.
  sorted_x_ranges = sorted(x_ranges, key=lambda x: x[0])
  merged_x_ranges = [sorted_x_ranges[0]]
  overlap_threshold = 0.2
  for x_range in sorted_x_ranges[1:]:
    overlapping_x_range = (
      max(x_range[0], merged_x_ranges[-1][0]),
      min(x_range[1], merged_x_ranges[-1][1])
    )
    overlapping_distance = float(overlapping_x_range[1] - overlapping_x_range[0])
    if (overlapping_distance / (x_range[1] - x_range[0]) > overlap_threshold or
      overlapping_distance / (merged_x_ranges[-1][1] - merged_x_ranges[-1][0]) > overlap_threshold) :
      merged_x_ranges[-1] = (
        min(x_range[0], merged_x_ranges[-1][0]),
        max(x_range[1], merged_x_ranges[-1][1]),
        min(x_range[2], merged_x_ranges[-1][2]),
        max(x_range[3], merged_x_ranges[-1][3])
      )
    else:
      saturation = np.sum(255 - img[x_range[2]:x_range[3],x_range[0]:x_range[1]])
      max_saturation = 255 * (x_range[3] - x_range[2]) * (x_range[1] - x_range[0])
      # Ignore noisy results (less than 10% saturation)
      if saturation > 0.1 * max_saturation:
          merged_x_ranges.append(x_range)

  # Determine the threshold for space characters: the width of the narrowest character
  x_range_widths = []
  for x_range in merged_x_ranges:
    x_range_widths.append(x_range[1] - x_range[0])
  space_threshold = np.quantile(np.array(x_range_widths), 0.5)

  # Insert space characters if the distance between adjacent characters is high
  chars = []
  space_indexes = []
  counter = 0
  for i in range(len(merged_x_ranges)):
    r1 = merged_x_ranges[i]
    chars.append(img[r1[2]:r1[3],r1[0]:r1[1]])
    counter += 1
    if draw:
      cv2.rectangle(img,(r1[0],r1[2]),(r1[1],r1[3]),(0,255,0),2)
    if i != len(merged_x_ranges) - 1:
      r2 = merged_x_ranges[i+1]
      if r2[0] - r1[1] > space_threshold:
        # chars.append(None)
        space_indexes.append(counter)
        counter += 1
        if draw:
          cv2.line(img, (int((r2[0] + r1[1])/2), 0), (int((r2[0] + r1[1])/2), img.shape[0]), (0,255,0), 2)

  if draw:
    return chars, space_indexes, img
  return chars, space_indexes

if __name__ == '__main__':
  input_f, output_dir = sys.argv[1:]
  input_name = input_f.split("/")[-1]
  input_quality = int(input_name.split(".")[1])
  
  img = cv2.imread(input_f)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # lines, drawn_img = extract_lines(img, draw=True)
  # out_name = 'page.{}.jpg'.format(input_quality)
  # cv2.imwrite(os.path.join(output_dir, out_name), drawn_img, (cv2.IMWRITE_JPEG_QUALITY, input_quality))
  lines = extract_lines(img)
  out_name = 'page.{}.jpg'.format(input_quality)
  cv2.imwrite(os.path.join(output_dir, out_name), img, (cv2.IMWRITE_JPEG_QUALITY, input_quality))
  for i, line in enumerate(lines):
    out_name = 'line{}.{}.jpg'.format(i, input_quality)
    cv2.imwrite(os.path.join(output_dir, out_name), line, (cv2.IMWRITE_JPEG_QUALITY, input_quality))
    chars,_,drawn_line = extract_chars(line, draw=True)
    out_name = 'line{}-drawn.{}.jpg'.format(i, input_quality)
    cv2.imwrite(os.path.join(output_dir, out_name), drawn_line, (cv2.IMWRITE_JPEG_QUALITY, input_quality))
    for j, char in enumerate(chars):
      if char is None:
        continue
      # out_name = 'line{}.char{}.{}.jpg'.format(i, j, input_quality)
      # cv2.imwrite(os.path.join(output_dir, out_name), char, (cv2.IMWRITE_JPEG_QUALITY, input_quality))
  
# # Finally show the img
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
