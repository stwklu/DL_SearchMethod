import cv2
import numpy as np
import random
import math

def drawRegion(img, corners, color=(0, 255, 0), thickness=1):
    # draw the bounding box specified by the given corners
    for i in range(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)

def to_homogeneous(corners):
    corners = corners.T
    return np.array([corners[0,], corners[1,], [1, 1, 1, 1]], dtype=int)

def mean_corners_error(delta_p_preds, corners_batch, corners_true_batch, rho=32):
    sum_err = 0
    for i in range(len(delta_p_preds)):
        delta_p = rho*delta_p_preds[i].reshape(4,2)
        corners = corners_batch[i]
        corners_true = corners_true_batch[i]
        corners_pred = corners + delta_p
        #print(corners_true.shape)
        #print(corners_pred.shape)
        #print(corners_true)
        #print(corners_pred)
        #print(corners_pred[0:2,])
        #corners_true = to_homogeneous(corners_true)
        #corners_pred = to_homogeneous(corners_pred)
        sum_err += math.sqrt(np.sum(np.square(corners_true - corners_pred)) / 4)
        #print(math.sqrt(np.sum(np.square(corners_true - corners_pred)) / 4))
    return sum_err