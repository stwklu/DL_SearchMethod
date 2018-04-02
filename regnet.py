import tensorflow as tf
import numpy as np

he_normal = tf.variance_scaling_initializer()

def Conv(inputs, kernel_size, strides, num_filters, weight_decay, name):
    '''
    Helper function to create a Conv2D layer
    '''
    with tf.variable_scope("conv2D_%s" % name):
        filter_shape = [kernel_size, kernel_size, inputs.get_shape()[3], num_filters]
        w = tf.get_variable(name='W', shape=filter_shape, 
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            initializer=he_normal)
        b = tf.get_variable(name='b', shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        out = tf.nn.relu(conv)
        print(name+":", out.get_shape())
    return out

class RegNet():
    def __init__(self, num_classes=3, imagenet_mean=[123.68, 116.779, 103.939], weight_decay=1e-4, 
                 enable_moving_average=False):
        # input tensors
        self.images_1 = tf.placeholder(tf.float32, [None, 128, 128], name="images_1")
        self.images_2 = tf.placeholder(tf.float32, [None, 128, 128], name="images_2")
        self.labels = tf.placeholder(tf.float32, [None, num_classes], name="labels")
        self.is_training = tf.placeholder(tf.bool)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        images_1 = tf.expand_dims(self.images_1, axis=-1)
        images_2 = tf.expand_dims(self.images_2, axis=-1)
        print("input tensors:")
        print(images_1.get_shape())
        print(images_2.get_shape())

        # General ImageNet Preprocessing
        #with tf.name_scope('zero_mean') as scope:
        #    mean = tf.constant(imagenet_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #    images_1 = self.images_1-mean
        #    images_2 = self.images_2-mean
        #    images = tf.concat(values=[images_1, images_2], axis=3)
        #    print(images.get_shape())

        # Standardization Preprocessing
        #with tf.name_scope('preprocess_alexnet') as scope:
        #    images = tf.map_fn(lambda image: tf.image.per_image_standardization(image), self.input_x)

        # Preprocessing for Inception
        with tf.name_scope('Inception_Preprocess') as scope:
            images = tf.concat(values=[images_1, images_2], axis=3)
            images /= 255.
            images -= 0.5
            images *=2

        print("Stacked images:", images.get_shape())
        # CNN Architecture
        conv1_1 = Conv(inputs=images, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv1_1")
        conv1_2 = Conv(inputs=conv1_1, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv1_2")
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")
        print('pool1', pool1.get_shape())

        conv2_1 = Conv(inputs=pool1, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv2_1")
        conv2_2 = Conv(inputs=conv2_1, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv2_2")
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")
        print('pool2', pool2.get_shape())
       
        conv3_1 = Conv(inputs=pool2, kernel_size=3, strides=1, num_filters=128, weight_decay=weight_decay, name="conv3_1")
        conv3_2 = Conv(inputs=conv3_1, kernel_size=3, strides=1, num_filters=128, weight_decay=weight_decay, name="conv3_2")
        pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")
        print('pool3', pool3.get_shape())

        conv4_1 = Conv(inputs=pool3, kernel_size=3, strides=1, num_filters=128, weight_decay=weight_decay, name="conv4_1")
        conv4_2 = Conv(inputs=conv4_1, kernel_size=3, strides=1, num_filters=128, weight_decay=weight_decay, name="conv4_2")

        shape = int(np.prod(conv4_2.get_shape()[1:]))
        flatten = tf.reshape(conv4_2, [-1, shape])
                                    
        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('W', [shape, 1024], initializer=he_normal,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable('b', [1024], initializer=tf.constant_initializer(0.0))
            fc1 = tf.matmul(flatten, w) + b
            fc1 = tf.nn.relu(fc1)
            print('fc1', fc1.get_shape())

        with tf.variable_scope('output'):
            w = tf.get_variable('w', [fc1.get_shape()[1], num_classes], initializer=he_normal)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
            self.predictions = tf.matmul(fc1, w) + b
            print('Output:', self.predictions.get_shape())

        # Calculate Mean Huber loss
        with tf.name_scope("loss"):
            losses = tf.losses.huber_loss(labels=self.labels, predictions=self.predictions)
            regularization_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            if enable_moving_average:
                total_loss = tf.reduce_mean(losses) + regularization_losses
                moving_averages = tf.train.ExponentialMovingAverage(0.9)
                moving_averages_op = moving_averages.apply([tf.reduce_mean(losses)] + [total_loss])
                with tf.control_dependencies([moving_averages_op]):
                    self.loss = tf.identity(total_loss)
            else:
                self.loss = tf.reduce_mean(losses) + regularization_losses
        
        with tf.name_scope("Euclidean_Loss"):
            self.l2_loss = tf.norm(self.labels-self.predictions)

#regnet = RegNet()