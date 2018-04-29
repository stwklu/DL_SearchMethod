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
from synthetic.perturbate_dof import *

from generator import *

TEST_SAMPLES = 5000

warp_func = 'hom'
rho = 64.
batch_size = 64

# Mean corners error
def mean_corners_err(y_true, y_pred):
    return K.mean(64*K.sqrt(K.sum(K.square(K.reshape(y_pred, (-1,4,2)) - K.reshape(y_true, (-1,4,2))),\
        axis=-1, keepdims=True)), axis=1)

def eval(rho=64, warp_func='hom', model_file='homography_model_compiled.json', weights_file='checkpoints/homography_model_weights.h5'):
    print("\nTesting:")
    json_file = open(model_file, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='mse', metrics=[mean_corners_err])
    model.load_weights(weights_file)
    print('Model loaded.')

    score = model.evaluate_generator(generator('./dataset/test', batch_size),
                                     steps=int(np.floor(TEST_SAMPLES/batch_size)))
    print("\nloss:", score[0])
    print("\nmean average corner error:", score[1])

def predict_image(img_fname, rho=32, patch_size=128, warp_func='hom', 
                  model_file='homography_model_compiled.json', weights_file='checkpoints/homography_model_weights.h5', visualization=1):
    img = cv2.imread(img_fname)
    height, width = 240, 320
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
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
    #eval(weights_file='hom_epoch_12.h5')
    predict_image(img_fname='test_images/000000111006.jpg', warp_func='hom', visualization=1)