import cv2
import numpy as np
import random

# Homography (8 dof)
def hom(frame, patch_size, rho): 
    height, width = frame.shape[:2]
    #print("image:", height, width)
    
    # Get corners
    y, x = np.random.randint(rho, height - rho - patch_size), np.random.randint(rho, width - rho - patch_size)
    corners = np.array([[x,y], 
                    [x + patch_size, y], 
                    [patch_size + x, patch_size + y], 
                    [x, y + patch_size]], dtype=np.float32)
    
    delta_p = np.float32([[np.random.randint(-rho, rho), np.random.randint(-rho, rho)],
                         [np.random.randint(-rho, rho), np.random.randint(-rho, rho)],
                         [np.random.randint(-rho, rho), np.random.randint(-rho, rho)],
                         [np.random.randint(-rho, rho), np.random.randint(-rho, rho)]])
    #delta_p = np.array([[ -9,-29],[ -2, 28], [ 16, 3], [-23, -8]])
    perturbed_corners = delta_p + corners
    #print()
    #print(corners.shape)
    #print(perturbed_corners.shape)
    #print(corners)
    #print(perturbed_corners)
    #print(delta_p)
    #print()

    # Get homography
    H = cv2.getPerspectiveTransform(np.float32(perturbed_corners), np.float32(corners))
    perturbed_frame = frame.copy()
    # Perturbe frame
    perturbed_frame = cv2.warpPerspective(perturbed_frame, H, (width, height), flags=cv2.INTER_CUBIC)

    # Crop patch here
    patch_1 = frame[y:y + patch_size, x:x + patch_size]
    patch_2 = perturbed_frame[y:y + patch_size, x:x + patch_size]

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