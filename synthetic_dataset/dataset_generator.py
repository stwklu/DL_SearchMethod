import cv2
import numpy as np
import math
import time
import random
from tracker_utils import *
from matplotlib import pyplot as plt

# ground truth location drawn in green
ground_truth_color = (0, 255, 0)
# tracker location drawn in red
prediction_color = (0, 0, 255)

def fix_size(boundary, max_boundary):
    if boundary < 0:
        boundary = 0
    if boundary > max_boundary:
        boundary = max_boundary-1
    return boundary

def get_bbox(frame, corners):
    corners = corners.astype(int)
    min_x = np.amin(corners[0])
    max_x = np.amax(corners[0])
    min_y = np.amin(corners[1])
    max_y = np.amax(corners[1])

    #print(min_x, max_x, min_y, max_y)
    #print(new_corners)
    min_x = fix_size(min_x, frame.shape[1])
    max_x = fix_size(max_x, frame.shape[1])
    min_y = fix_size(min_y, frame.shape[0])
    max_y = fix_size(max_y, frame.shape[0])

    return min_x, max_x, min_y, max_y

def synthetic_dataset(frame, corners, num_samples=10):
    # Normal distributions used for delta_p
    tx_samples = np.round(np.random.normal(loc=0, scale=5, size=num_samples))
    ty_samples = np.round(np.random.normal(loc=0, scale=5, size=num_samples))
    scale_samples = np.absolute(np.random.normal(loc=1, scale=0.15, size=num_samples))
    scale_samples[scale_samples > 1.15] = 1.15
    scale_samples[scale_samples < 0.85] = 0.85

    previous_image_samples = []
    previous_corners_samples = []
    image_samples = []
    delta_p_samples = []
    corners_samples = []
    for i in range(num_samples):
        # Warp function and new corners
        W = np.array([[scale_samples[i], 0, tx_samples[i]], [0, scale_samples[i], ty_samples[i]], [0, 0, 1]])
        new_corners = np.around(np.dot(W, corners)).astype(int)

        #mce = math.sqrt(np.sum(np.square(corners - new_corners)) / 4)
        #print(mce)

        # Crop by old ground truth corners
        min_x, max_x, min_y, max_y = get_bbox(frame, corners)
        crop_prev_frame = frame[min_y:max_y, min_x:max_x]
        crop_prev_frame = cv2.resize(crop_prev_frame, (224, 224))
        previous_image_samples.append(crop_prev_frame)

        # Crop by synthetic corners
        try:
            min_x, max_x, min_y, max_y = get_bbox(frame, new_corners)
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
        previous_corners_samples.append(corners)
        corners_samples.append(new_corners)
    return np.array(image_samples), np.array(delta_p_samples), np.array(previous_image_samples), np.array(previous_corners_samples), np.array(corners_samples)

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
    all_delta_ps = []
    all_prev_images = []
    all_prev_corners = []
    all_corners = []
    # Main loop
    for i in range(no_of_frames):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if i==0:
            old_frame = frame
        if not ret:
            print("Initial frame could not be read")
            sys.exit(0)
        corners = np.array([np.append(ground_truths[i, 0:2], [1]),
                            np.append(ground_truths[i, 2:4], [1]),
                            np.append(ground_truths[i, 4:6], [1]),
                            np.append(ground_truths[i, 6:8], [1])]).T
        images, delta_ps, prev_images, prev_corners, corners = synthetic_dataset(frame, corners)
        all_images.append(images)
        all_delta_ps.append(delta_ps)
        all_prev_images.append(prev_images)
        all_prev_corners.append(prev_corners)
        all_corners.append(corners)

    cap.release()
    all_images = np.concatenate(all_images)
    all_delta_ps = np.concatenate(all_delta_ps)
    all_prev_images = np.concatenate(all_prev_images)
    all_prev_corners = np.concatenate(all_prev_corners)
    all_corners = np.concatenate(all_corners)

    # Random shuffle the entire set
    indices = np.random.permutation(np.arange(len(all_images)))
    all_images = all_images[indices]
    all_delta_ps = all_delta_ps[indices]
    all_prev_images = all_prev_images[indices]
    all_prev_corners = all_prev_corners[indices]
    all_corners = all_corners[indices]
    print(all_images.shape)
    print(all_delta_ps.shape)

    #drawRegion(old_frame, all_prev_corners[0], ground_truth_color, thickness=2)
    #drawRegion(old_frame, all_corners[0], prediction_color, thickness=2)
    #plt.imshow(all_images[0])
    #plt.imshow(old_frame)
    #plt.show()
    print(all_delta_ps[0])
    np.save('images.npy', all_images)
    np.save('prev_images.npy', all_prev_images)
    np.save('delta_ps.npy', all_delta_ps)
    np.save('prev_corners.npy', all_prev_corners)
    np.save('corners.npy', all_corners)