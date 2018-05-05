import keras, cv2, h5py, json, math
import numpy as np
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import losses, regularizers
import keras.backend as K
from matplotlib import pyplot as plt

from models import *
from synthetic.image_utils import *
from synthetic.perturbate_dof_old import *

from generator import *

_SAMPLES_PER_ARCHIVE = 8000
TRAIN_SAMPLES = 64 * _SAMPLES_PER_ARCHIVE
VAL_SAMPLES = 1000
TEST_SAMPLES = 5000

warp_func = 'hom'
batch_size = 64

def eval(warp_func='hom', model_file='homography_model_compiled.json', weights_file='checkpoints/hom_epoch_12.h5'):
    # Loss Function using SMSE
    def euclidean_l2(y_true, y_pred):
        # Multiply preds by a factor of 2 described in paper
        # prediction rho in range [-32, 32] while 
        # rho in test set is [-64, 64]
        return K.sqrt(K.sum(K.square(2*y_pred - y_true), axis=-1, keepdims=True))

    # Mean corners error
    def mean_corners_err(y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(K.reshape(2*y_pred, (-1,4,2)) - K.reshape(y_true, (-1,4,2))),\
                      axis=-1, keepdims=True)), axis=1)

    json_file = open(model_file, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss=euclidean_l2, metrics=[mean_corners_err])
    model.load_weights(weights_file)
    print('\nModel loaded.')

    print("Testing on test set...")
    score = model.evaluate_generator(generator('./dataset/test_'+warp_func, batch_size),
                                     steps=int(np.floor(TEST_SAMPLES/batch_size)), verbose=1)
    print("Loss on test set:", score[0])
    print("Test Mean Corner Error:", score[1])


def eval_train(warp_func='hom', model_file='homography_model_compiled.json', weights_file='checkpoints/hom_epoch_12.h5'):
    # Loss Function using SMSE
    def euclidean_l2(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

    # Mean corners error
    def mean_corners_err(y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(K.reshape(y_pred, (-1,4,2)) - K.reshape(y_true, (-1,4,2))),\
                      axis=-1, keepdims=True)), axis=1)

    json_file = open(model_file, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss=euclidean_l2, metrics=[mean_corners_err])
    model.load_weights(weights_file)
    print('\nModel loaded.')

    print("\nTesting on validation set...")
    score = model.evaluate_generator(generator('./dataset/val_'+warp_func, batch_size),
                                     steps=int(np.floor(VAL_SAMPLES/batch_size)), verbose=1)
    print("Loss on validation set:", score[0])
    print("Val Mean Corner Error:", score[1])

    print("\nTesting on train set...")
    score = model.evaluate_generator(generator('./dataset/train_'+warp_func, batch_size),
                                     steps=int(np.floor(TRAIN_SAMPLES/batch_size)), verbose=1)
    print("Loss on train set:", score[0])
    print("Train Mean Corner Error:", score[1])

def predict_image(img_fname, rho=32, patch_size=128, warp_func='hom', 
                  model_file='homography_model_compiled.json', weights_file='checkpoints/hom_epoch_12.h5', visualization=1):
    img = cv2.imread(img_fname)
    height, width = 240, 320
    img = center_crop(img, (width, height))
    #img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    if warp_func=='hom':
        patch_1, patch_2, delta_p, corners, perturbed_corners, perturbed_img = hom(img, patch_size, rho)
    elif warp_func=='trans':
        patch_1, patch_2, delta_p, corners, perturbed_corners, perturbed_img = trans(img, patch_size, rho)

    # Load models
    json_file = open('homography_model_compiled.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(optimizer=SGD(lr=0.005), loss='mse', metrics=[mean_corners_err])
    model.load_weights(weights_file)
    model.summary()
    
    # Prepare patch
    patch_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2GRAY)
    patch_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2GRAY)
    patch_1 = (np.float32(patch_1) - 127.5)/127.5
    patch_2 = (np.float32(patch_2) - 127.5)/127.5
    patch = np.dstack((patch_1, patch_2))

    # Prediction
    # Predict delta p
    H_4point = model.predict(np.array([patch]))[0]
    H_4point *= rho
    H_4point = H_4point.astype(int)
    err_delta_p = math.sqrt(np.sum(np.square(H_4point - delta_p)) / 4)
    
    # Update corners
    H_4point = H_4point.reshape(4,2)
    predicted_corners = corners + H_4point
    err_corners = math.sqrt(np.sum(np.square(perturbed_corners - predicted_corners)) / 4)


    print('\nMean corner error between corners:', err_corners)
    print('Mean corner error between delta p:', err_delta_p)

    print('\ndelta p predicted:', H_4point)
    print('delta p true:', delta_p.reshape(4,2))
    print('predicted corner:', predicted_corners)
    print('perturbed corner:', perturbed_corners)

    # Visualization
    if visualization:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        perturbed_img = cv2.cvtColor(perturbed_img, cv2.COLOR_BGR2RGB)
        drawRegion(img, to_homogeneous(corners), thickness=2)
        drawRegion(perturbed_img, to_homogeneous(perturbed_corners), (0, 255, 0), thickness=2)
        drawRegion(perturbed_img, to_homogeneous(predicted_corners), (255, 0, 0), thickness=2)
        plt.subplot(1,2,1)
        plt.imshow(img, interpolation='bicubic')
        plt.subplot(1,2,2)
        plt.imshow(perturbed_img, interpolation='bicubic')
        plt.show()

if __name__=='__main__':
    eval()
    eval_train()
    #predict_image(img_fname='test_images/uncropped.jpg', warp_func='hom', visualization=1)