import cv2
import numpy as np
import math
import time
import random
from tracker_utils import *

# Some default parameters
# thickness of the bounding box lines drawn on the image
thickness = 2
# ground truth location drawn in green
ground_truth_color = (0, 255, 0)
# tracker location drawn in red
prediction_color = (0, 0, 255)
# DOF for the parametric representation of warp
dof = 3

# For synthetic dataset
# an offset added to generate a range of bbox
offset = 5


def synthetic_dataset(current_frame, current_corners, next_corners, no_samples=1000):
    difference = next_corners - current_corners
    for i in range(no_samples):
        scale = 0
        tx = 0
        ty = 0
        
    return


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

    tracking_errors = []
    tracking_fps = []
    # Main loop
    for i in range(no_of_frames):
        ret, frame = cap.read()
        if not ret:
            print("Initial frame could not be read")
            sys.exit(0)
        corners_true = np.array([np.append(ground_truths[i, 0:2], [1]),
                            np.append(ground_truths[i, 2:4], [1]),
                            np.append(ground_truths[i, 4:6], [1]),
                            np.append(ground_truths[i, 6:8], [1])]).T

        if i+1 < no_of_frames:
            corners_pred = np.array([np.append(ground_truths[i+1, 0:2], [1]),
                                np.append(ground_truths[i+1, 2:4], [1]),
                                np.append(ground_truths[i+1, 4:6], [1]),
                                np.append(ground_truths[i+1, 6:8], [1])]).T
        else:
            corners_pred = corners_true

        # update the tracker with the current frame
        #print(corners_true.shape)
        #print(corners_true)
        # W, corners_pred = ist_transformation(delta_p_samples[idx, :], corners_true)
        #synthetic_dataset(current_frame=None, current_corners=corners_true, next_corners=corners_pred, no_samples=1000)
        
        

        # Compute scores
        current_mce = math.sqrt(np.sum(np.square(corners_true - corners_pred)) / 4)
        tracking_errors.append(current_mce)

        cv2.putText(frame, "{:5.2f}".format(current_mce), (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        drawRegion(frame, corners_true, ground_truth_color, thickness=2)
        drawRegion(frame, corners_pred, prediction_color, thickness=2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()