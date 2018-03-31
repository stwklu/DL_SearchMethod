import cv2
import numpy as np
import math
import time
import random
from tracker_utils import *
from matplotlib import pyplot as plt

def fix_size(boundary, max_boundary):
    if boundary < 0:
        boundary = 0
    if boundary > max_boundary:
        boundary = max_boundary-1
    return boundary

def synthetic_dataset(frame, corners, num_samples=100):
    tx_samples = np.round(np.random.normal(loc=0, scale=5, size=num_samples))
    ty_samples = np.round(np.random.normal(loc=0, scale=5, size=num_samples))
    scale_samples = np.absolute(np.random.normal(loc=1, scale=0.2, size=num_samples))
    scale_samples[scale_samples > 1.2] = 1.2
    scale_samples[scale_samples < 0.8] = 0.8

    image_samples = []
    delta_p_samples = []
    for i in range(num_samples):
        W = np.array([[scale_samples[i], 0, tx_samples[i]], [0, scale_samples[i], ty_samples[i]], [0, 0, 1]])
        new_corners = np.around(np.dot(W, corners)).astype(int)
        min_x = np.amin(new_corners[0])
        max_x = np.amax(new_corners[0])
        min_y = np.amin(new_corners[1])
        max_y = np.amax(new_corners[1])

        print(min_x, max_x, min_y, max_y)
        #print(new_corners)
        min_x = fix_size(min_x, frame.shape[1])
        max_x = fix_size(max_x, frame.shape[1])
        min_y = fix_size(min_y, frame.shape[0])
        max_y = fix_size(max_y, frame.shape[0])

        print(min_x, max_x, min_y, max_y)
        try:
            crop_frame = frame[min_y:max_y, min_x:max_x]
            crop_frame = cv2.resize(crop_frame, (224, 224))
        except:
            print(corners)
            print(W)
            print(new_corners)
            plt.imshow(frame)
            plt.show()
            
            exit()
        
        image_samples.append(crop_frame)
        delta_p_samples.append(np.array([tx_samples[i], ty_samples[i], scale_samples[i]]))
    return np.array(image_samples), np.array(delta_p_samples)

#frame = cv2.imread('cmt_lemming/frame00001.jpg')
#corners = np.array([[39. ,104.,104.,39.], [198.,198.,308.,308.], [1., 1., 1., 1.]])
#images, labels = synthetic_dataset(frame, corners)

if __name__ == '__main__':
    # Sequence to be loaded
    seq_name = 'cmt_lemming'
    src_fname = seq_name + '/frame%05d.jpg'
    ground_truth_fname = seq_name + '.txt'

    # OpenCV settings
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print('The video file ', src_fname, ' could not be opened')
        sys.exit()

    # Load ground truth bbox
    # ground_truths to be of shape (no_of_frames, 4)
    # up_left_coordinate, up_right, down_right, down_left
    ground_truths = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truths.shape[0]
    print('no_of_frames: ', no_of_frames)

    all_images = []
    all_labels = []
    # Main loop
    for i in range(no_of_frames):
        ret, frame = cap.read()
        if not ret:
            print("Initial frame could not be read")
            sys.exit(0)
        corners = np.array([np.append(ground_truths[i, 0:2], [1]),
                            np.append(ground_truths[i, 2:4], [1]),
                            np.append(ground_truths[i, 4:6], [1]),
                            np.append(ground_truths[i, 6:8], [1])]).T
        images, labels = synthetic_dataset(frame, corners)
        all_images.append(images)
        all_labels.append(labels)

    cap.release()
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    print(all_images.shape)
    print(all_labels.shape)
    plt.imshow(all_images[0])
    plt.show()
    print(all_labels[0])