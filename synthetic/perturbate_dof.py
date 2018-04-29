import cv2
import numpy as np
import random
from image_utils import *

# Homography (8 dof)
def hom(frame, patch_size=128, rho=24, rho_2=16, training=True, 
        synthesis_tracking=True, crop_method='avg'): 
    # Discard image if it's too small
    # Image will be resized to match the DeepHomography settings
    if frame.shape < (240, 320):
        return None, None
    if training:
        width, height = 320, 240
    else:
        width, height = 640, 480
    frame = scale_down(frame, (width, height))
    frame = center_crop(frame, (width, height))

    # Get corners
    y, x = np.random.randint(rho, height - rho - patch_size), np.random.randint(rho, width - rho - patch_size)
    corners = np.array([[x,y], 
                    [x + patch_size, y], 
                    [patch_size + x, patch_size + y], 
                    [x, y + patch_size]], dtype=np.float32)

    # Synthesis the tracking scenario where 1st corner is also warped
    if synthesis_tracking:
        delta_p_init = np.float32([[np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)],
                            [np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)],
                            [np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)],
                            [np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)]])
        corners += delta_p_init
        if crop_method == 'avg':
            x_min, x_max, y_min, y_max = avg_corner(corners)
        elif crop_method == 'min':
            x_min, x_max, y_min, y_max = minmax_corner(corners)
        else:
            raise Exception('unsupported cropping method')
    else:
        x_min, x_max, y_min, y_max = x, x + patch_size, y, y + patch_size
    
    # Random perturb corner
    delta_p = np.float32([[np.random.randint(-rho, rho), np.random.randint(-rho, rho)],
                         [np.random.randint(-rho, rho), np.random.randint(-rho, rho)],
                         [np.random.randint(-rho, rho), np.random.randint(-rho, rho)],
                         [np.random.randint(-rho, rho), np.random.randint(-rho, rho)]])
    perturbed_corners = delta_p + corners

    # Get homography
    H = cv2.getPerspectiveTransform(np.float32(perturbed_corners), np.float32(corners))
    # Perturbe frame
    perturbed_frame = cv2.warpPerspective(frame.copy(), H, (width, height), flags=cv2.INTER_CUBIC)

    # Crop patch here
    patch_1 = frame[y_min:y_max, x_min:x_max]
    patch_2 = perturbed_frame[y_min:y_max, x_min:x_max]

    return patch_1, patch_2, delta_p.reshape(8), corners, perturbed_corners, perturbed_frame

# Translation (2 dof)
def trans(frame, patch_size, rho, x, y):
    height, width = frame.shape[:2]
    # Get corners
    y, x = np.random.randint(rho, height - rho - patch_size), np.random.randint(rho, width - rho - patch_size)
    corners = np.array([[x,y], 
                    [x + patch_size, y], 
                    [patch_size + x, patch_size + y], 
                    [x, y + patch_size]], dtype=np.float32)
    delta_p = np.float32(np.array([np.random.randint(-rho, rho), np.random.randint(-rho, rho)]))
    perturbed_corners = delta_p + corners
    # Apply inverse M on image???
    inv_delta_p = delta_p
    M = np.float32([[1,0,inv_delta_p[0]],[0,1,inv_delta_p[1]]])
    perturbed_frame = cv2.warpAffine(frame.copy(), M, (width, height), flags=cv2.INTER_CUBIC)

    # Crop patch here
    patch_1 = frame[y:y + patch_size, x:x + patch_size]
    patch_2 = perturbed_frame[y:y + patch_size, x:x + patch_size]

    return patch_1, patch_2, delta_p, corners, perturbed_corners, perturbed_frame

# Affine 6 dof
def affine(frame, patch_size, rho): 
    height, width = frame.shape[:2]
    # define corners of image patch
    y, x = random.randint(rho, height - rho - patch_size), random.randint(rho, width - rho - patch_size)
    corners = np.array([[x,y], 
                    [x + patch_size, y], 
                    [patch_size + x, patch_size + y], 
                    [x, y + patch_size]], dtype=np.float32)
    delta_p = np.float32([[np.random.randint(-rho, rho),np.random.randint(-rho, rho)]
                         [np.random.randint(-rho, rho),np.random.randint(-rho, rho)],
                         [np.random.randint(-rho, rho),np.random.randint(-rho, rho)]])
    perturbed_corners = delta_p + corners
    M = cv2.getAffineTransform(perturbed_corners, corners)
    perturbed_frame = cv2.warpAffine(frame.copy(), M, (width, height), flags=cv2.INTER_CUBIC)

    # Crop patch here
    patch_1 = frame[y:y + patch_size, x:x + patch_size]
    patch_2 = perturbed_frame[y:y + patch_size, x:x + patch_size]

    return patch_1, patch_2, delta_p.reshape(6), corners, perturbed_corners, perturbed_frame