import glob
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import os
import shutil
import uuid

from perturbate_dof import *
from image_utils import *

# ground truth location drawn in green
warp_func = 'hom'
ground_truth_color = (0, 255, 0)
visualization = 1

# Folder separation
num_samples_per_archive = 8000

# Synthetic settings
num_train_samples = 42000
num_val_samples = 1024
num_test_samples = 5000

def generate_patch(frame_fname, dataset_path, patch_size, rho, no_sample, warp_func='trans', training=True, to_grayscale=True):
    frame = cv2.imread(frame_fname)
    if training:
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    else:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if warp_func=='hom':
        patch_1, patch_2, delta_p, corners, perturbed_corners, perturbed_frame = hom(frame, patch_size, rho)
    elif warp_func=='trans':
        patch_1, patch_2, delta_p, corners, perturbed_corners, perturbed_frame = trans(frame, patch_size, rho)
    if patch_1 is None or patch_2 is None or np.isnan(patch_1).any() or np.isnan(patch_2).any():
        return False, None, None, None, None

    # Save images to pre-defined dataset path
    if not training:
        patch_1 = cv2.resize(patch_1, (128, 128), interpolation=cv2.INTER_AREA)
        patch_2 = cv2.resize(patch_2, (128, 128), interpolation=cv2.INTER_AREA)
    
    if to_grayscale:
        patch_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2GRAY)
        patch_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2GRAY)

    patch = np.stack((patch_1, patch_2), axis=-1)
    delta_p = np.array(delta_p, dtype=np.float32)

    if visualization:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        perturbed_frame = cv2.cvtColor(perturbed_frame, cv2.COLOR_BGR2RGB)
        drawRegion(frame, to_homogeneous(corners), thickness=2)
        drawRegion(frame, to_homogeneous(perturbed_corners), (255, 0, 0), thickness=2)
        drawRegion(perturbed_frame, to_homogeneous(corners), (255, 0, 0), thickness=2)
        print(delta_p)
        print(np.array(delta_p).shape)
        print(os.path.join(dataset_path, str(no_sample)+"_"+frame_fname.split('/')[-1][:-4]+'_patch.npy'))
        print(os.path.join(dataset_path, str(no_sample)+"_"+frame_fname.split('/')[-1][:-4]+'_delta_p.npy'))
        patch_1 = cv2.cvtColor(patch_1, cv2.COLOR_GRAY2RGB)
        patch_2 = cv2.cvtColor(patch_2, cv2.COLOR_GRAY2RGB)
        plt.subplot(1,2,1)
        plt.title('Original image')
        plt.imshow(frame)
        plt.subplot(1,2,2)
        plt.title('After applying homography')
        plt.imshow(perturbed_frame)
        plt.show()
        plt.subplot(1,2,1)
        plt.title('patch_1')
        plt.imshow(patch_1)
        plt.subplot(1,2,2)
        plt.title('patch_2')
        plt.imshow(patch_2)
        plt.show()
        exit()

    return True, patch, delta_p, corners, perturbed_corners

def generate_dataset(num_samples, patch_size, rho, dataset='train', training=True, num_synthesis=1):
    print('start generating', dataset, 'samples...')
    # Folder to read in images
    images = np.array(glob.glob('./'+dataset+'2014/*.jpg'))
    # Folder to be saved
    dataset_path = os.path.join(os.getcwd(), dataset+'_'+warp_func)
    print(dataset_path)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)

    # !!!TOO MANY, NEED TO SAVE IMAGES AS SINGLE FILES ON DISK!!!
    num_samples_generated = 0
    num_synthesis_done = 0
    i = 0
    if not training:
        height, width = 480, 640
    else:
        height, width = 240, 320
    
    patches_samples = []
    delta_p_samples = []
    while num_samples_generated < num_samples*num_synthesis:
        # Generate init coordiante here
        succeed, patch, delta_p, corners, perturbed_corners = generate_patch(frame_fname=images[i], warp_func=warp_func, rho=rho, no_sample=num_samples_generated+1,
                                     patch_size=patch_size, dataset_path=dataset_path, training=training)

        if succeed:
            num_samples_generated += 1
            num_synthesis_done += 1
            patches_samples.append(patch)
            delta_p_samples.append(delta_p)
            if num_synthesis_done >= num_synthesis:
                i += 1
                num_synthesis_done = 0

        if i >= len(images):
            i = 0
        if not succeed:
            print(images[i])
        if len(delta_p_samples) >= num_samples_per_archive:
            # Save as npz files here
            name = str(uuid.uuid4())
            pack = os.path.join(dataset_path, name + '.npz')
            with open(pack, 'wb') as f:
                np.savez(f, images=np.stack(patches_samples), offsets=np.stack(delta_p_samples))
            print('bundled:', name)
            patches_samples = []
            delta_p_samples = []

    if len(delta_p_samples) > 0:
        # Save as npz files here
        name = str(uuid.uuid4())
        pack = os.path.join(dataset_path, name + '.npz')
        with open(pack, 'wb') as f:
            np.savez(f, images=np.stack(patches_samples), offsets=np.stack(delta_p_samples))
        print('bundled:', name)

    print('success.')

if __name__ == '__main__':
    generate_dataset(num_train_samples, patch_size=128, rho=32, dataset='train', num_synthesis=12)
    #generate_dataset(num_val_samples, patch_size=128, rho=32, dataset='val', training=False, num_synthesis=1)
    #generate_dataset(num_test_samples, patch_size=256, rho=64, dataset='test', training=False, num_synthesis=1)
