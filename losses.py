import keras.backend as K

'''
Homography
'''
# Loss Function using SMSE
def SMSE(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

# Mean corner error
def mean_corner_err(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(K.reshape(y_pred, (-1,4,2)) - K.reshape(y_true, (-1,4,2))),\
        axis=-1, keepdims=True)), axis=1)