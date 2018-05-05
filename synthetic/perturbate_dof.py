import cv2
import numpy as np
import random
from synthetic.image_utils import *

def homography(img, width=320, height=240, patch_size=128, training=True,
               rho_1=32, rho_2=16, synthesize_tracking=False, crop_method='min_max'):
    # Synthesize the tracking scenario when 1st corner is also possibly warped
    if synthesize_tracking:
        y, x = np.random.randint(rho_1+rho_2, height - rho_1 - rho_2 - patch_size), np.random.randint(rho_1+rho_2, width - rho_1 - rho_2 - patch_size)
        #x, y = (random.randint(24 + rho_1 + rho_2, patch_size + rho_1 + rho_2), 24 + rho_1 + rho_2)
        delta_p_init = np.float32([[np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)],
                                  [np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)],
                                  [np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)],
                                  [np.random.randint(-rho_2, rho_2), np.random.randint(-rho_2, rho_2)]])
        corners += delta_p_init
        if crop_method == 'avg':
            x_min, x_max, y_min, y_max = avg_corner(corners)
        elif crop_method == 'min_max':
            x_min, x_max, y_min, y_max = minmax_corner(corners)
        else:
            raise Exception('unsupported cropping method')
    else:
        y, x = np.random.randint(rho_1, height - rho_1 - patch_size), np.random.randint(rho_1, width - rho_1 - patch_size)
        #x, y = (random.randint(24 + rho_1, patch_size + rho_1), 24 + rho_1)
        x_min, x_max, y_min, y_max = x, x + patch_size, y, y + patch_size

    # Choose top-left corner of patch (assume 0,0 is top-left of image)
    # Restrict points to within 24-px from the border
    corners = np.array([[x,y], 
                       [x + patch_size, y], 
                       [patch_size + x, patch_size + y], 
                       [x, y + patch_size]], dtype=np.float32)

    # Random perturb corner
    delta_p = np.float32([[np.random.randint(-rho_1, rho_1), np.random.randint(-rho_1, rho_1)],
                         [np.random.randint(-rho_1, rho_1), np.random.randint(-rho_1, rho_1)],
                         [np.random.randint(-rho_1, rho_1), np.random.randint(-rho_1, rho_1)],
                         [np.random.randint(-rho_1, rho_1), np.random.randint(-rho_1, rho_1)]])
    perturbed_corners = delta_p + corners

    # Get homography
    H = cv2.getPerspectiveTransform(np.float32(perturbed_corners), np.float32(corners))
    # Perturbe img
    perturbed_img = cv2.warpPerspective(img, H, (width, height), flags=cv2.INTER_CUBIC)

    # Crop patch here
    patch_1 = img[y_min:y_max, x_min:x_max]
    patch_2 = perturbed_img[y_min:y_max, x_min:x_max]

    if synthesize_tracking:
        patch_1 = cv2.resize(patch_1, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        patch_2 = cv2.resize(patch_2, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    if not training:
        patch_1 = cv2.resize(patch_1, (128, 128), interpolation=cv2.INTER_AREA)
        patch_2 = cv2.resize(patch_2, (128, 128), interpolation=cv2.INTER_AREA)

    return patch_1, patch_2, delta_p.reshape(-1), corners

# Translation (2 dof)
def translation(img, width=320, height=240, patch_size=128, training=True, rho=32):
    # Get corners
    #x, y = (random.randint(24 + rho_1, patch_size + rho_1), 24 + rho_1)
    y, x = np.random.randint(rho, height - rho - patch_size), np.random.randint(rho, width - rho - patch_size)
    corners = np.array([[x,y], 
                    [x + patch_size, y], 
                    [patch_size + x, patch_size + y], 
                    [x, y + patch_size]], dtype=np.float32)
    delta_p = np.float32(np.array([np.random.randint(-rho, rho), np.random.randint(-rho, rho)]))
    perturbed_corners = delta_p + corners
    # Apply inverse M on image
    M = np.float32([[1, 0, -delta_p[0]],[0, 1, -delta_p[1]]])
    perturbed_img = cv2.warpAffine(img.copy(), M, (width, height), flags=cv2.INTER_CUBIC)

    # Crop patch here
    patch_1 = img[y:y + patch_size, x:x + patch_size]
    patch_2 = perturbed_img[y:y + patch_size, x:x + patch_size]

    if not training:
        patch_1 = cv2.resize(patch_1, (128, 128), interpolation=cv2.INTER_AREA)
        patch_2 = cv2.resize(patch_2, (128, 128), interpolation=cv2.INTER_AREA)

    return patch_1, patch_2, delta_p, corners