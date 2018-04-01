from dataset_generator import *

frame = cv2.imread('cmt_lemming/frame00001.jpg')
corners = np.array([[39. ,104.,104.,39.], [198.,198.,308.,308.], [1., 1., 1., 1.]])
images, labels = synthetic_dataset(frame, corners)
plt.imshow(images[0])
plt.show()
print(labels[0])