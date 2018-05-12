'''
Eval Script for HomographyNet on synthetic dataset
'''
import keras
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import losses, regularizers
import keras.backend as K
import numpy as np
from absl import flags
import cv2
import h5py
import math
import sys
import datetime
from matplotlib import pyplot as plt

from synthetic.image_utils import *
from synthetic.perturbate_dof_old import *
from generator import *

# Parameters settings
# Dataset Parameters
flags.DEFINE_integer("samples_per_archive", 8000, "Number of samples per training npz archive (default: 8000)")
flags.DEFINE_integer("num_val_samples", 1000, "Number of validation samples")
flags.DEFINE_integer("num_test_samples", 5000, "Number of test samples")
flags.DEFINE_integer("factor", 2, "Factor for delta_p, default 2 for image of 640 x 480 ")
flags.DEFINE_string('warp_func', 'hom', 'Warping parameter to be used')

# Eval Parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")


FLAGS = flags.FLAGS
FLAGS(sys.argv)
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value.value))
print("")

_SAMPLES_PER_ARCHIVE = 8000
TRAIN_SAMPLES = 64 * _SAMPLES_PER_ARCHIVE

def eval(weights_file='checkpoints/hom_epoch_12.h5'):
    # Loss Function using SMSE
    def SMSE(y_true, y_pred):
        # Multiply preds by a factor of 2 described in paper
        # prediction rho in range [-32, 32] while 
        # rho in test set is [-64, 64]
        return K.sqrt(K.sum(K.square(FLAGS.factor*y_pred - y_true), axis=-1, keepdims=True))

    # Mean corners error
    def mean_corner_err(y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(K.reshape(FLAGS.factor*y_pred, (-1,4,2)) - K.reshape(y_true, (-1,4,2))),\
                      axis=-1, keepdims=True)), axis=1)

    # Init models here
    json_file = open("homography_net_"+FLAGS.warp_func+"_model.json", 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss=SMSE, metrics=[mean_corner_err])
    model.load_weights(weights_file)
    print('\nModel loaded.')

    # Testing...
    print("Testing on test set...")
    score = model.evaluate_generator(generator('./dataset/test_'+FLAGS.warp_func, FLAGS.batch_size),
                                     steps=int(np.floor(FLAGS.num_test_samples/FLAGS.batch_size)), verbose=1)
    print("Loss on test set:", score[0])
    print("Test Mean Corner Error:", score[1])


def eval_train(weights_file='checkpoints/hom_epoch_12.h5'):
    # Loss Function using SMSE
    def SMSE(y_true, y_pred):
        # Multiply preds by a factor of 2 described in paper
        # prediction rho in range [-32, 32] while 
        # rho in test set is [-64, 64]
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

    # Mean corners error
    def mean_corner_err(y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(K.reshape(y_pred, (-1,4,2)) - K.reshape(y_true, (-1,4,2))),\
                      axis=-1, keepdims=True)), axis=1)

    # Init models here
    json_file = open("homography_net_"+FLAGS.warp_func+"_model.json", 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss=SMSE, metrics=[mean_corner_err])
    model.load_weights(weights_file)
    print('\nModel loaded.')

    print("\nTesting on validation set...")
    score = model.evaluate_generator(generator('./dataset/val_'+FLAGS.warp_func, FLAGS.batch_size),
                                     steps=int(np.floor(FLAGS.num_val_samples/FLAGS.batch_size)), verbose=1)
    print("Loss on validation set:", score[0])
    print("Val Mean Corner Error:", score[1])

    print("\nTesting on train set...")
    score = model.evaluate_generator(generator('./dataset/train_'+FLAGS.warp_func, FLAGS.batch_size),
                                     steps=int(np.floor((64*FLAGS.samples_per_archive)/FLAGS.batch_size)), verbose=1)
    print("Loss on train set:", score[0])
    print("Train Mean Corner Error:", score[1])

if __name__=='__main__':
    eval()
    #eval_train()
    #eval_image(img_fname='test_images/uncropped.jpg', warp_func='hom', visualization=1)