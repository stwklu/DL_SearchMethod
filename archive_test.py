import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
get some infos from npz archive
'''

path = '90b9b677-2c20-42d9-8ac5-8f98de3d0b82.npz'
archive = np.load(path)
print('keys:', archive.files)

images = archive['images']
offsets = archive['offsets']
print('images.shape:', images.shape)
print('offsets.shape:', offsets.shape)

print('sample patches:', images[40].shape)
print('H_4pts:', offsets[40])
plt.subplot(1,2,1)
plt.title('patch 1')
plt.imshow(images[25,:,:,0], 'gray')
plt.subplot(1,2,2)
plt.title('patch 2')
plt.imshow(images[25,:,:,1], 'gray')
plt.show()