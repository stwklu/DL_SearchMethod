'''
Training Script for HomographyNet
'''
import keras, cv2, h5py, json, math
import numpy as np
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import losses, regularizers
import keras.backend as K

from models import *
from generator import *
from losses import *

_SAMPLES_PER_ARCHIVE = 8000
TRAIN_SAMPLES = 64 * _SAMPLES_PER_ARCHIVE
VAL_SAMPLES = 1000

warp_func = 'hom'
num_classes = 8
batch_size = 64
num_epochs = 12

# Reduce learning rate around every 30000 iteration
def halve_lr(epoch):
    if epoch > 8:
        lr = 5e-5
    elif epoch > 4:
        lr = 5e-4
    else:
        lr = 5e-3
    print('learning rate', lr)
    return lr

def train():
    # Init Keras Model here
    model = homography_net(num_classes=num_classes)
    model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss=SMSE, metrics=[mean_corner_err])
    model_json = model.to_json()
    with open("homography_model_compiled.json","w") as json_file:
        json_file.write(model_json)                    # Save model architecture

    # Trainer
    learning_rate = LearningRateScheduler(halve_lr)
    checkpointer = ModelCheckpoint(filepath="./checkpoints/"+warp_func+"_epoch_{epoch:02d}.h5", period=1,
                                   verbose=1, save_best_only=True, mode='min', monitor='loss')
    model.fit_generator(generator('./dataset/train_'+warp_func, batch_size), 
                        steps_per_epoch=int(np.floor(TRAIN_SAMPLES/batch_size)), 
                        epochs=num_epochs, 
                        verbose=1,
                        validation_data=generator('./dataset/val_'+warp_func, batch_size), 
                        validation_steps=int(np.floor(VAL_SAMPLES/batch_size)),
                        callbacks=[checkpointer, learning_rate])
    model.save_weights('checkpoints/homography_model_weights.h5')  # Save model weights
    print("Model saved.")

if __name__=='__main__':
    train()