import cv2
import numpy as np
import math
import time
import random
from tracker_utils import *

import tensorflow as tf

# Prediction Functions
def get_warp(delta_p, dof=3):
    if dof==3:
        return np.array([[delta_p[2], 0, delta_p[0]], [0, delta_p[2], delta_p[1]], [0, 0, 1]])
    return None

def warp_corners(W, corners):
    return np.around(np.dot(W, corners)).astype(int)

def get_delta_p_prediction(x1_batch, x2_batch):
    """
    Get prediction on a test set
    """
    feed_dict = {logits.images_1: x1_batch, 
                 logits.images_2: x2_batch, 
                 logits.is_training: False,
                 logits.dropout_keep_prob: 1.0}
    preds = sess.run(logits.predictions, feed_dict)
    return preds

# Some default parameters
# Parameters settings

# Model Hyperparameters
tf.app.flags.DEFINE_integer("dof", 3, "Degree of freedom (DOF) for warp function")
tf.app.flags.DEFINE_string("checkpoint_path", "checkpoints", "Path for the checkpoint to be used.")

FLAGS = tf.app.flags.FLAGS
print("-"*30)
print("Evaluation for Tracking")
print("-"*30)
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value.value))
print("")

# thickness of the bounding box lines drawn on the image
thickness = 2
# ground truth location drawn in green
ground_truth_color = (0, 255, 0)
# tracker location drawn in red
prediction_color = (0, 0, 255)

#sess = tf.Session()
#logits = RegNet(num_classes=FLAGS.dof)
#sess.run(tf.global_variables_initializer())

# Restore model
#checkpoint_dir = os.path.join(os.path.curdir, FLAGS.checkpoint_path)
#saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
#print("Model restored!")


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

        # Ground truth corners
        corners_true = np.array([np.append(ground_truths[i, 0:2], [1]),
                            np.append(ground_truths[i, 2:4], [1]),
                            np.append(ground_truths[i, 4:6], [1]),
                            np.append(ground_truths[i, 6:8], [1])]).T
        # Prediction corners
        if i==0:
            # Initial corners
            corners_pred = np.array([np.append(ground_truths[0, 0:2], [1]),
                                    np.append(ground_truths[0, 2:4], [1]),
                                    np.append(ground_truths[0, 4:6], [1]),
                                    np.append(ground_truths[0, 6:8], [1])]).T
        else:
            #delta_p = get_delta_p_prediction(patch_1, patch_2)
            #W = get_warp(delta_p, dof=3)
            #corners_pred = warp_corners(W, corners_pred)
            corners_pred = corners_true
    
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