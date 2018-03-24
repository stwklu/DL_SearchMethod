import cv2
import numpy as np
from tracker_utils import *

# Some default parameters
# thickness of the bounding box lines drawn on the image
thickness = 2
# ground truth location drawn in green
ground_truth_color = (0, 255, 0)
# tracker location drawn in red
prediction_color = (0, 0, 255)


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

    for i in range(no_of_frames):
        ret, frame = cap.read()
        if not ret:
            print("Initial frame could not be read")
            sys.exit(0)
        corners = np.array([ground_truths[i, 0:2],
                            ground_truths[i, 2:4],
                            ground_truths[i, 4:6],
                            ground_truths[i, 6:8]]).T
        drawRegion(frame, corners, ground_truth_color, thickness=2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()