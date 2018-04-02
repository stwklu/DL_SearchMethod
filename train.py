import tensorflow as tf
import numpy as np
import datetime
import data_helper
import os

from regnet import *

# Parameters settings

# Dataset Parameters
tf.app.flags.DEFINE_float("dev_sample_percentage", 0.2, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("weight_decay", 1e-4, "Weight decay rate for L2 regularization (default: 5e-4)")

# Training Parameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Starter Learning Rate (default: 1e-3)")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_boolean("enable_moving_average", True, "Enable usage of Exponential Moving Average (default: True)")

FLAGS = tf.app.flags.FLAGS
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value.value))
print("")

# Loading database here
print("Loading database...")
images_1, images_2, labels = data_helper.load_dataset()
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(labels)))
x1_train, x1_dev = images_1[:dev_sample_index], images_1[dev_sample_index:]
x2_train, x2_dev = images_2[:dev_sample_index], images_2[dev_sample_index:]
y_train, y_dev = labels[:dev_sample_index], labels[dev_sample_index:]
num_batches_per_epoch = int((len(x1_train)-1)/FLAGS.batch_size) + 1
print(x1_train.shape, x2_train.shape, y_train.shape)
print(x1_dev.shape, x2_dev.shape, y_dev.shape)
print("Success!")

# Initialize tf graph
sess = tf.Session()
logits = RegNet(num_classes=y_dev.shape[1],
enable_moving_average=FLAGS.enable_moving_average,
weight_decay=FLAGS.weight_decay)

# Optimizer and LR Decay
global_step = tf.train.create_global_step()
learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
#optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
gradients, variables = zip(*optimizer.compute_gradients(logits.loss))
#gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

# Initialize Graph
sess.run(tf.global_variables_initializer())

# Output directory for Tensorflow models and summaries
out_dir = os.path.curdir

# Tensorboard
def add_gradient_summaries(grads_and_vars):
    grad_summaries = []
    for grad, var in grads_and_vars:
        if grad is not None:
            grad_hist_summary = tf.summary.histogram(var.op.name + "/gradient", grad)
            grad_summaries.append(grad_hist_summary)
    return grad_summaries
hist_summaries = []
for var in tf.trainable_variables():
    hist_hist_summary = tf.summary.histogram(var.op.name + "/histogram", var)
    hist_summaries.append(hist_hist_summary)
hist_summaries_merged = tf.summary.merge(hist_summaries)
grad_summaries = add_gradient_summaries(zip(gradients, variables))
grad_summaries_merged = tf.summary.merge(grad_summaries)

# Summaries for loss and euclidean loss
loss_summary = tf.summary.scalar("loss", logits.loss)
l2_loss_summary = tf.summary.scalar("Euclidean_loss", logits.l2_loss)
# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, l2_loss_summary, hist_summaries_merged, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# Saver
# Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.join(os.path.curdir, 'checkpoints')
print("Saving models to {}".format(checkpoint_dir))
checkpoint_prefix = os.path.join(checkpoint_dir, "regnet_model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

# Train Step and Dev Step
def train_step(x1_batch, x2_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {logits.images_1: x1_batch, 
                 logits.images_2: x2_batch, 
                 logits.labels: y_batch, 
                 logits.is_training: True,
                 logits.dropout_keep_prob: FLAGS.dropout_keep_prob}
    _, step, loss, l2_loss = sess.run([train_op, global_step, logits.loss, logits.euclidean_loss], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Step {}, Epoch {}, Loss {:g}, l2 Loss {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss, l2_loss/FLAGS.batch_size))
    if step % FLAGS.evaluate_every==0:
        summaries = sess.run(train_summary_op, feed_dict)
        train_summary_writer.add_summary(summaries, global_step=step)

def dev_step(x1_batch, x2_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {logits.images_1: x1_batch, 
                 logits.images_2: x2_batch, 
                 logits.labels: y_batch, 
                 logits.is_training: False,
                 logits.dropout_keep_prob: 1.0}
    loss, preds, l2_loss = sess.run([logits.loss, logits.predictions, logits.euclidean_loss], feed_dict)
    return loss, preds, l2_loss/FLAGS.batch_size

# Training loop. For each batch...
min_loss = 1000
min_step = 0
# Generate batches
train_batches = data_helper.batch_iter(list(zip(x1_train, x2_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
for train_batch in train_batches:
    x1_batch, x2_batch, y_batch = zip(*train_batch)
    train_step(x1_batch, x2_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)

    # deving loop
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        sum_loss = 0
        sum_l2_loss = 0
        i = 0
        dev_batches = data_helper.batch_iter(list(zip(x1_dev, x2_dev, y_dev)), FLAGS.batch_size, 1, shuffle=False)
        for dev_batch in dev_batches:
            x1_dev_batch, x2_dev_batch, y_dev_batch = zip(*dev_batch)
            dev_loss, preds, l2_loss = dev_step(x1_dev_batch, x2_dev_batch, y_dev_batch)
            sum_loss += dev_loss
            sum_l2_loss += l2_loss
            i += 1
        avg_loss = sum_loss/i
        avg_l2_loss = sum_l2_loss/i
        time_str = datetime.datetime.now().isoformat()
        print("{}: Evaluation Summary, Epoch {}, Avg. Loss {:g}, l2 Loss {:g}".format(
              time_str, int(current_step//num_batches_per_epoch)+1, avg_loss, avg_l2_loss))
        if avg_l2_loss < min_loss:
            min_loss = avg_l2_loss
            min_step = current_step
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved current model checkpoint with min loss to {}".format(path))
        print("{}: Current Min Avg. Loss {:g} in Iteration {}".format(time_str, min_loss, min_step))