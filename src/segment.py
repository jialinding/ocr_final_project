import numpy as np
import cv2
import sys
import os

# Takes image, returns list of images (each representing a row of text)
# Assumes img is grayscale
def extract_lines(img):
	# Remove left and right margins
	# Threshold to determine if column is not blank: 1%
	y_sum = np.sum(255-img, 0)
	y_blank = (y_sum > img.shape[0] * 255 * 0.01).astype(np.int)
	nonzero_y = np.nonzero(y_blank)[0]
	img = img[:, nonzero_y[0]:nonzero_y[-1]+1]

	x_sum = np.sum(255-img, 1)

	# Threshold to determine if row is not blank: 2%
	x_blank = np.concatenate([
		[0],
		(x_sum > img.shape[1] * 255 * 0.02).astype(np.int),
		[0]
	])
	x_diff = x_blank[1:] - x_blank[:-1]
	row_starts = np.where(x_diff == 1)[0]
	row_ends = np.where(x_diff == -1)[0]
	rows = []
	for i in range(row_starts.shape[0]):
		# Ignore noisy results (rows that have height less than 10 pixels)
		if row_ends[i] - row_starts[i] < 10:
			continue

		# print(i, row_starts[i], row_ends[i])
		# cv2.line(image, (-10, row_starts[i]), (3000, row_starts[i]), (0,255,0), 1)
		# cv2.line(image, (-10, row_ends[i]), (3000, row_ends[i]), (0,0,255), 1)
		rows.append(img[row_starts[i]:row_ends[i], :])
	return rows

# Takes image representing row of text, returns list of images (each reepreseting a character)
# Assumes img is grayscale
# Spaces are represented as "None" in the returned list
# Assumes that characters are separated by white space,
# and each character is one connected piece of non-white space
# Based on https://stackoverflow.com/questions/10964226/how-to-convert-an-image-into-character-segments
def extract_chars(img):
	# smooth the image to avoid noises
	# img = cv2.medianBlur(img,5)

	# Apply adaptive threshold
	thresh = cv2.adaptiveThreshold(img,255,1,1,11,2)

	# Find the contours
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		return []

	# Extract the x-ranges of each contour
	x_ranges = []
	for cnt in contours:
	    x,y,w,h = cv2.boundingRect(cnt)
	    # Ignore noisy results (bounding rectangle is less than 5 pixels in width or height)
	    if w < 5 or h < 5:
	    	continue

	    x_ranges.append((x, x+w))

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
				max(x_range[1], merged_x_ranges[-1][1])
			)
		else:
			saturation = np.sum(255 - img[:,x_range[0]:x_range[1]])
			max_saturation = 255 * img.shape[0] * (x_range[1] - x_range[0])
			# Ignore noisy results (less than 5% saturation)
			if saturation > 0.05 * max_saturation:
			    merged_x_ranges.append(x_range)

	# Determine the threshold for space characters: the width of the narrowest character
	x_range_widths = []
	for x_range in merged_x_ranges:
		x_range_widths.append(x_range[1] - x_range[0])
	space_threshold = np.quantile(np.array(x_range_widths), 0.1)

	# Insert space characters if the distance between adjacent characters is high
	chars = []
	for i in range(len(merged_x_ranges)):
		r1 = merged_x_ranges[i]
		chars.append(img[:,r1[0]:r1[1]])
		# cv2.rectangle(img,(r1[0],int(img.shape[0] * 0.1)),(r1[1],int(img.shape[0] * 0.9)),(0,255,0),2)
		if i != len(merged_x_ranges) - 1:
			r2 = merged_x_ranges[i+1]
			if r2[0] - r1[1] > space_threshold:
				chars.append(None)
				# cv2.line(img, (int((r2[0] + r1[1])/2), 0), (int((r2[0] + r1[1])/2), img.shape[0]), (0,255,0), 1)

	return chars

input_f, output_dir = sys.argv[1:]
input_name = input_f.split("/")[-1]
input_quality = int(input_name.split(".")[1])

image = cv2.imread(input_f)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lines = extract_lines(image)
for i, line in enumerate(lines):
	out_name = 'line{}.{}.jpg'.format(i, input_quality)
	cv2.imwrite(os.path.join(output_dir, out_name), line, (cv2.IMWRITE_JPEG_QUALITY, input_quality))
	chars = extract_chars(line)
	for j, char in enumerate(chars):
		if char is None:
			continue
		out_name = 'line{}.char{}.{}.jpg'.format(i, j, input_quality)
		cv2.imwrite(os.path.join(output_dir, out_name), char, (cv2.IMWRITE_JPEG_QUALITY, input_quality))

# # Finally show the image
# cv2.imshow('img',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
