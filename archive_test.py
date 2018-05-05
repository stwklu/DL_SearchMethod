import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
get some infos from npz archive
'''

path = 'ced71c20-93e8-4b83-8f17-78a39d25c799.npz'
archive = np.load(path)
print('keys:', archive.files)

images = archive['images']
offsets = archive['offsets']
print('images.shape:', images.shape)
print('offsets.shape:', offsets.shape)

print('sample patches:', images[10].shape)
print('H_4pts:', offsets[10])
plt.subplot(1,2,1)
plt.title('patch 1')
plt.imshow(images[10,:,:,0], 'gray')
plt.subplot(1,2,2)
plt.title('patch 2')
plt.imshow(images[10,:,:,1], 'gray')
plt.show()