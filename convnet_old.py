import tensorflow as tf
import numpy as np
from alexnet_v2 import *

class ConvNet():
    def __init__(self, num_classes, imagenet_mean=[123.68, 116.779, 103.939], weight_decay=1e-4, 
                 enable_moving_average=False):
        # input tensors
        self.images_1 = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images_1")
        self.images_2 = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images_2")
        self.labels = tf.placeholder(tf.float32, [None, num_classes], name="labels")
        self.is_training = tf.placeholder(tf.bool)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # General ImageNet Preprocessing
        with tf.name_scope('zero_mean') as scope:
            mean = tf.constant(imagenet_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images_1 = self.images_1-mean
            images_2 = self.images_2-mean
            images = tf.concat(values=[images_1, images_2], axis=3)
            print(images.get_shape())

        # Standardization Preprocessing
        #with tf.name_scope('preprocess_alexnet') as scope:
        #    images = tf.map_fn(lambda image: tf.image.per_image_standardization(image), self.input_x)

        # Preprocessing for MobileNet
        #with tf.name_scope('MobileNet_Preprocess') as scope:
            #images = self.input_x/255.
            #images -= 0.5
            #images *=2

        # AlexNet
        with slim.arg_scope(alexnet_v2_arg_scope(weight_decay)):
            logits, _ = alexnet_v2(inputs=images, num_classes=num_classes, is_training=self.is_training)
        print(logits.get_shape())

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = logits
            self.euclidean_loss = tf.norm(self.labels-logits)
            losses = tf.losses.huber_loss(labels=self.labels, predictions=logits)
            regularization_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            if enable_moving_average:
                total_loss = tf.reduce_mean(losses) + regularization_losses
                moving_averages = tf.train.ExponentialMovingAverage(0.9)
                moving_averages_op = moving_averages.apply([tf.reduce_mean(losses)] + [total_loss])
                with tf.control_dependencies([moving_averages_op]):
                    self.loss = tf.identity(total_loss)
            else:
                self.loss = tf.reduce_mean(losses) + regularization_losses

#logits = ConvNet(num_classes=3)