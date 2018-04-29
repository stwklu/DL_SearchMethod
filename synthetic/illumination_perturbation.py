import glob
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import os
import shutil

# Parameters
sigma_range = 100
intensity = 1000

# Folder to read in images
images = np.array(glob.glob("./train2014/*.jpg"))
# Folder to be saved
dataset_path = os.path.join(os.getcwd(), "synthetic")

# 2d gaussian distribution for illumination perturbation
def Gaussian2d(intensity, mu_x, mu_y, sigma_x, sigma_y, width, height):
    size = 1000
    x = np.linspace(0, width, num=width)
    y = np.linspace(0, height, num=height)

    x, y = np.meshgrid(x, y)
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-((x-mu_x)**2/(2*sigma_x**2) + (y-mu_y)**2/(2*sigma_y**2))))

    intensity = intensity / z[mu_y][mu_x]
    z = z * intensity
    return z


def illumination(frame):
	height, width = frame.shape[:2]

	#Modeling illumination variations with 2D Gaussian functions
	# Parameters

	mu_x = int(random.uniform(0, width))
	mu_y = int(random.uniform(0, height))
	sigma_x = random.uniform(0, sigma_range)
	sigma_y = random.uniform(0, sigma_range)

	#mu_x, mu_y, sigma_x, sigma_y = 200, 200, 50, 50

	gaussian_light = Gaussian2d(intensity, mu_x, mu_y, sigma_x, sigma_y, width, height)
	#gaussian_light = gaussian_light.astype(np.uint8)
	frame = np.add(frame, gaussian_light)

	return frame, gaussian_light.astype(np.uint8)

# Test driver
def main():
	global images, dataset_path

	indices = np.random.permutation(np.arange(len(images)))
	images = images[indices]
	frame = cv2.imread(images[0])
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	new_frame, gaussian_light = illumination(frame)

	plt.imshow(gaussian_light, 'gray', interpolation='bicubic')
	plt.title('Gaussian 2D Lighting')
	plt.show()

	# Visualize result
	plt.subplot(1,2,1)
	plt.title('Before Gaussian Lighting')
	plt.imshow(frame,'gray')
	plt.subplot(1,2,2)
	plt.title('After Gaussian Lighting')
	plt.imshow(new_frame, 'gray')
	plt.show()

if __name__ == '__main__':
	main()