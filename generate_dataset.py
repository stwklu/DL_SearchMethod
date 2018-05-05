import sys
import random
import glob
import os.path
import uuid
import shutil

from queue import Queue, Empty
from threading import Thread

import numpy as np
import cv2
from synthetic.image_utils import *
from synthetic.perturbate_dof import *

def process_image(image_path, num_samples=1, training=True, synthesize_tracking=False):
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Separate training and testing
    if training:
        width, height = 320, 240
        patch_size = 128
        rho_1 = 32
        rho_2 = 16
        target_size = (width, height)
        img = scale_down(img, target_size)
    else:
        width, height = 640, 480
        patch_size = 256
        rho_1 = 64
        rho_2 = 32
        target_size = (width, height)
        img = scale_up(img, target_size)

    img = center_crop(img, target_size)

    image_pairs = []
    offsets = []
    #orig_points = []
    #perturbed_points = []
    while len(offsets) < num_samples:
        # Warping function goes here
        #patch_1, patch_2, delta_p, corners = homography(img, width=width, height=height, patch_size=patch_size, training=training,
        #                        rho_1=rho_1, rho_2=rho_2, synthesize_tracking=synthesize_tracking, crop_method='min_max')
        patch_1, patch_2, delta_p, corners = translation(img, width=width, height=height, patch_size=patch_size, 
                                                         training=training, rho=rho_1)
        
        try:
            d = np.stack((patch_1, patch_2), axis=-1)
        except ValueError:
            continue
        image_pairs.append(d)
        offsets.append(delta_p)
        #orig_points.append(orig)
        #perturbed_points.append(perturbed)
    print('done:', image_path)
    return image_pairs, offsets


class Worker(Thread):
   def __init__(self, input_queue, output_queue, num_samples, training, synthesize_tracking):
       Thread.__init__(self)
       self.input_queue = input_queue
       self.output_queue = output_queue
       self.num_samples = num_samples
       self.training = training
       self.synthesize_tracking = synthesize_tracking

   def run(self):
       while True:
           img_path = self.input_queue.get()
           if img_path is None:
               break
           output = process_image(img_path, self.num_samples, self.training, self.synthesize_tracking)
           self.input_queue.task_done()
           if output is not None:
               self.output_queue.put(output)


def pack(outdir, image_pairs, offsets):
    name = str(uuid.uuid4())
    pack = os.path.join(outdir, name + '.npz')
    with open(pack, 'wb') as f:
        np.savez(f, images=np.stack(image_pairs), offsets=np.stack(offsets))
    print('bundled:', name)


def bundle(queue, outdir, num_samples_per_archive=8000):
    image_pairs = []
    offsets = []
    #orig_points = []
    #perturbed_points = []
    while True:
        try:
            d, o = queue.get(timeout=10)
        except Empty:
            break
        image_pairs.extend(d)
        offsets.extend(o)
        #orig_points.extend(orig)
        #perturbed_points.extend(perturbed)

        if len(image_pairs) >= num_samples_per_archive:
            pack(outdir, image_pairs, offsets)
            image_pairs = []
            offsets = []
        queue.task_done()

    if image_pairs:
        pack(outdir, image_pairs, offsets)


def generate_dataset(num_images=64000, training=True, dataset='train', warp_func='hom', num_samples=8, synthesize_tracking=False):
    # 512000, 8, 8000
    input_dir = '/home/ubuntu/sdd/'+dataset+'2014'
    #input_dir = './train2014'
    output_dir = './dataset/'+dataset+'_'+warp_func
    print('saving dataset in', output_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Create a queue to communicate with the worker threads
    input_queue = Queue()
    output_queue = Queue()

    num_workers = 8
    workers = []
    # Create worker threads
    for i in range(num_workers):
        worker = Worker(input_queue, output_queue, num_samples, training, synthesize_tracking)
        worker.start()
        workers.append(worker)

    width, height = 320, 240

    count = 0
    for i in glob.iglob(os.path.join(input_dir, '*.jpg')):
        # Discard if image is too small
        if cv2.imread(i).shape[:2] >= (height, width):
            input_queue.put(i)
            count += 1
        if count >= num_images:
            break

    bundle(output_queue, output_dir)

    input_queue.join()
    for i in range(num_workers):
        input_queue.put(None)
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    generate_dataset(training=True, dataset='train', synthesize_tracking=False)
    generate_dataset(num_images=1000, training=True, dataset='val', num_samples=1, synthesize_tracking=False)
    generate_dataset(num_images=5000, training=False, dataset='test', num_samples=1, synthesize_tracking=False)
