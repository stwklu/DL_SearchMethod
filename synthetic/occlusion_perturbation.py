'''
 Reference: https://github.com/ahmedassal/Visual_Servoing_from_Deep_Neural_Networks/

 Pseudo-code:
	1- randomly select images from MS-COCO dataset
	2- segment image using slic
	3- randomly select segment
	4- randomly select position for the insertion of segment pixels
	5- insert segment pixels into target image

'''

import glob
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import os
import shutil
# Segmentation tool
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# Global variables
n_segs = 20
sigma = 5

# Folder to read in images
images = np.array(glob.glob("./train2014/*.jpg"))
# Folder to be saved
dataset_path = os.path.join(os.getcwd(), "synthetic")


# TODO
def save_image(frame):
	pass

def occlusion(frame):
	global Images
	inp_img_idx = 0

	# 1- randomly select images from MS-COCO dataset
	img = cv2.imread(images[np.random.randint(0, len(images))])

	# 2- segment image using slic
	segs = slic(img_as_float(img), n_segments=n_segs, sigma=sigma)


	# 3- randomly select segment, and extract its pixles
	mask = extract_mask(img, segs)
	seg_cropped, mask_cropped = crop_image(img, mask)

	# TO DO
	# 4- randomly select position for the insertion of segment pixels
	img_loc_start, img_loc_end  = generate_random_loc(inp_img_idx, images, seg_cropped)

	# 5- insert segment pixels into target image
	mask_cropped_f = cv2.normalize(mask_cropped.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	frame = overlay_image(frame, seg_cropped, img_loc_start, mask_cropped_f)

	save_image(frame)
	# Increment indices

	return frame


# randomly select position for the insertion of segment pixels
def generate_random_loc(inp_img_idx, images, seg_cropped):
	img_cropped_size = np.array(np.shape(seg_cropped)[:2])
	inp_img = cv2.imread(images[inp_img_idx])
	inp_img_size = np.array(np.shape(inp_img)[:2])
	img_valid_loc = np.subtract(inp_img_size, img_cropped_size)
	inp_img_loc_x = np.random.random_integers(high=img_valid_loc[1], low=0)
	inp_img_loc_y = np.random.random_integers(high=img_valid_loc[0], low=0)
	inp_img_loc_start = np.array([inp_img_loc_y, inp_img_loc_x])
	inp_img_loc_end = np.add(inp_img_loc_start, img_cropped_size)

	return inp_img_loc_start, inp_img_loc_end


# Extract pixel from segment
def crop_image(image, mask):
	dst = cv2.bitwise_and(image, image, mask=mask)
	bbox = cv2.boundingRect(mask)
	img_corp = dst[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
	mask_corp = mask[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]

	return img_corp, mask_corp


# randomly select segment
def extract_mask(image, segs):
	found=False
	while not found:
		seg = np.random.choice(np.unique(segs), size=1, replace=False)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segs == seg] = 255
		image_area = np.shape(image)[0] * np.shape(image)[1]
		mask_area = sum(sum(mask[:]))
		if 0.05*image_area < mask_area <= 0.2 * image_area:
			found = True
	return mask


"""Overlay img_overlay on top of img at the position specified by
pos and blend using alpha_mask.
Alpha mask must contain values within the range [0, 1] and be the
same size as img_overlay.
"""
def overlay_image(img, img_overlay, pos, alpha_mask):
	x, y = pos

	# Image ranges
	y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
	x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

	# Overlay ranges
	y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
	x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

	# Exit if nothing to do
	if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
		return

	channels = img.shape[2]

	alpha = alpha_mask[y1o:y2o, x1o:x2o]
	alpha_inv = 1.0 - alpha

	for c in range(channels):
		img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
		                        alpha_inv  * img[y1:y2, x1:x2, c])
	return img


# Test driver
def main():
	global images, dataset_path

	indices = np.random.permutation(np.arange(len(images)))
	images = images[indices]
	frame = cv2.imread(images[0])
	perturb_frame = occlusion(frame.copy())
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	perturb_frame = cv2.cvtColor(perturb_frame, cv2.COLOR_BGR2RGB)

	# Visualize result
	plt.subplot(1,2,1)
	plt.title('Before Occlusion')
	plt.imshow(frame)
	plt.subplot(1,2,2)
	plt.title('After Occlusion')
	plt.imshow(occlusion(perturb_frame))
	plt.show()

if __name__ == '__main__':
	main()