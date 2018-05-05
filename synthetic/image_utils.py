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

def scale_down(img, target_size):
    src_height, src_width = img.shape[:2]
    src_ratio = src_height/src_width
    target_width, target_height = target_size
    if src_ratio < target_height/target_width:
        dst_size = (int(np.round(target_height/src_ratio)), target_height)
    else:
        dst_size = (target_width, int(np.round(target_width*src_ratio)))
    return cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)

def scale_up(img, target_size):
    src_height, src_width = img.shape[:2]
    src_ratio = src_height/src_width
    target_width, target_height = target_size
    if src_ratio < target_height/target_width:
        dst_size = (int(np.round(target_height/src_ratio)), target_height)
    else:
        dst_size = (target_width, int(np.round(target_width*src_ratio)))
    return cv2.resize(img, dst_size, interpolation=cv2.INTER_CUBIC)


def center_crop(img, target_size):
    target_width, target_height = target_size
    # Note the reverse order of width and height
    height, width = img.shape[:2]
    x = int(np.round((width - target_width)/2))
    y = int(np.round((height - target_height)/2))
    return crop(img, (x, y), target_size)

def crop(img, origin, size):
    width, height = size
    x, y = origin
    return img[y:y + height, x:x + width]

def rect2corner(rect):
    return np.array([[rect[0], rect[2]], [rect[1], rect[2]], [rect[1], rect[3]], [rect[0], rect[3]]])

def minmax_corner(corners):
    '''
    Crop by the maximum of x, y coordinates
    '''
    x_max = np.max(corners[:,0])
    x_min = np.min(corners[:,0])
    y_max = np.max(corners[:,1])
    y_min = np.min(corners[:,1])
    #print(x_min, x_max, y_min, y_max)
    return int(x_min), int(x_max), int(y_min), int(y_max)

def avg_corner(corners, patch_size=128):
    '''
    Crop by the average x, y of given corners
    '''
    x_mean = np.mean(corners[:,0])
    y_mean = np.mean(corners[:,1])
    #print(np.mean(corners, axis=0))
    return int(x_mean-np.floor(patch_size/2)), int(x_mean+np.ceil(patch_size/2)), int(y_mean-np.floor(patch_size/2)), int(y_mean+np.ceil(patch_size/2))

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