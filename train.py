'''
Training Script for HomographyNet
'''
import keras
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras import losses, regularizers
import keras.backend as K
import numpy as np
from absl import flags
import cv2
import h5py
import math
import sys
import datetime

from models import *
from generator import *
from losses import *

# Parameters settings
# Dataset Parameters
flags.DEFINE_integer("samples_per_archive", 8000, "Number of samples per training npz archive (default: 8000)")
flags.DEFINE_integer("num_val_samples", 1000, "Number of validation samples")
flags.DEFINE_integer("num_classes", 8, "Number of classes")
flags.DEFINE_string('warp_func', 'hom', 'Warping parameter to be used')

# Model Hyperparameters
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training Parameters
flags.DEFINE_float("init_learning_rate", 0.005, "Initial Learning Rate (default: 5e-3)")
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 12, "Number of training epochs (default: 12)")

FLAGS = flags.FLAGS
FLAGS(sys.argv)
#FLAGS._parse_flags()
print("\nParameters:")
print("-"*20)
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value.value))
print("")

# Reduce learning rate around every 30000 iteration
def halve_lr(epoch):
    if epoch > 8:
        lr = 5e-5
    elif epoch > 4:
        lr = 5e-4
    else:
        lr = 5e-3
    time_str = datetime.datetime.now().isoformat()
    print("{}: learning rate {:g} at epoch {}".format(time_str, lr, epoch+1))
    return lr

def train():
    # Init Keras Model here
    model = homography_net(num_classes=FLAGS.num_classes, dropout_keep_prob=FLAGS.dropout_keep_prob)
    model.compile(optimizer=SGD(lr=FLAGS.init_learning_rate, momentum=0.9), loss=SMSE, metrics=[mean_corner_err])

    model_json = model.to_json()
    with open("homography_net_"+FLAGS.warp_func+"_model.json","w") as json_file:
        json_file.write(model_json)                    # Save model architecture
    time_str = datetime.datetime.now().isoformat()
    print("{}: Model saved as json.".format(time_str))

    # Trainer
    learning_rate = LearningRateScheduler(halve_lr)
    checkpointer = ModelCheckpoint(filepath="./checkpoints/"+FLAGS.warp_func+"_epoch_{epoch:02d}.h5", period=1,
                                   verbose=1, save_best_only=True, mode='min', monitor='loss')
    tensorboard = TensorBoard(log_dir='./logs', batch_size=FLAGS.batch_size, write_images=True)
    model.fit_generator(generator('./dataset/train_'+FLAGS.warp_func, FLAGS.batch_size), 
                        steps_per_epoch=int(np.floor((64*FLAGS.samples_per_archive)/FLAGS.batch_size)), 
                        epochs=FLAGS.num_epochs, 
                        verbose=1,
                        validation_data=generator('./dataset/val_'+FLAGS.warp_func, FLAGS.batch_size), 
                        validation_steps=int(np.floor(FLAGS.num_val_samples/FLAGS.batch_size)),
                        callbacks=[checkpointer, tensorboard, learning_rate])
    # Save the final trained model here
    model.save_weights('checkpoints/homography_model_weights.h5')  # Save model weights
    time_str = datetime.datetime.now().isoformat()
    print("{}: Training complete, model saved.".format(time_str))

if __name__=='__main__':
    train()