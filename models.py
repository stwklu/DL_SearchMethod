import keras
from keras.models import Model
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Activation, BatchNormalization

def homography_net(input_tensor=None, num_classes=8, dropout_keep_prob=0.5, kernel_initializer='glorot_uniform'):
    img_input = Input(shape=(128, 128, 2), name='patch')

    x = Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', name='conv1_1', kernel_initializer=kernel_initializer)(img_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', name='conv1_2', kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')(x)

    x = Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', name='conv2_1', kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', name='conv2_2', kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', name='conv3_1', kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', name='conv3_2', kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', name='conv4_1', kernel_initializer=kernel_initializer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', name='conv4_2', kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dropout(dropout_keep_prob, name='dropout_1')(x)
    x = Dense(1024, kernel_initializer=kernel_initializer, activation='relu')(x)
    x = Dropout(dropout_keep_prob, name='dropout_2')(x)
    out = Dense(num_classes, kernel_initializer=kernel_initializer)(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs=inputs, outputs=out, name='homography_net')
    return model

if __name__ == "__main__":
    #from keras.utils import plot_model
    model = homography_net()
    model.summary()
    #plot_model(model, '../images/hom_net.png')
    model_json = model.to_json()
    with open("homography_model_compiled.json","w") as json_file:
        json_file.write(model_json)                    # Save model architecture
    time_str = datetime.datetime.now().isoformat()
    print("{}: Model saved as json.".format(time_str))