from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import parameters as par
import cv2
import random as rand
import time
import matplotlib.pyplot as plt
import scipy.signal as spsig
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')
# Display tensorflow info
tf.logging.set_verbosity(tf.logging.INFO)



def read_images():

    filenames = []
    seg_tensors = []
    img_tensors = []

    # Encode data
    seg_dir = './VOC2012-objects/SegmentationObject/'
    img_dir = './VOC2012-objects/JPEGImages/'
    for seg_file in os.listdir(seg_dir):
        filename = seg_file.split('.')[0]
        if len(filenames) > par.max_img:
            break
        if not os.path.isfile(img_dir + filename + '.jpg'):
            continue
        print("Now encoding image with filename {}".format(filename))
        filenames.append(filename)

        resized_seg = cv2.resize(src=cv2.imread(seg_dir + seg_file),
                                 dsize=(par.img_width, par.img_height))
        resized_img = cv2.resize(src=cv2.imread(img_dir + filename + '.jpg'),
                                 dsize=(par.img_width, par.img_height))

        resized_seg = resized_seg.reshape(par.img_width, par.img_height, 3)
        resized_img = resized_img.reshape(par.img_width, par.img_height, 3)

        resized_seg = np.asarray(resized_seg)
        resized_img = np.asarray(resized_img)

        resized_seg = resized_seg.astype(float) / 255
        resized_img = resized_img.astype(float) / 255

        seg_tensors.append(tf.convert_to_tensor(resized_seg, np.float32))
        img_tensors.append(tf.convert_to_tensor(resized_img, np.float32))

    total_img = len(filenames)
    n_train = par.train_percentage/100 * total_img

    train_img = []
    train_seg = []
    test_img = []
    test_seg = []

    for rand_idx in rand.sample(range(total_img), total_img):
        if len(train_img) < n_train:
            train_img.append(img_tensors[rand_idx])
            train_seg.append(seg_tensors[rand_idx])
        else:
            test_img.append(img_tensors[rand_idx])
            test_seg.append(seg_tensors[rand_idx])

    train_img = tf.data.Dataset.from_tensors(train_img)
    train_seg = tf.data.Dataset.from_tensors(train_seg)
    test_img = tf.data.Dataset.from_tensors(test_img)
    test_seg = tf.data.Dataset.from_tensors(test_seg)
    return train_img, train_seg, test_img, test_seg


def model_fn(features, labels, mode):
    """Model function."""

    # Input layer
    input_layer = tf.reshape(features["x"],
                             [-1, par.img_width, par.img_height, 3])

    dense = tf.layers.dense(inputs=input_layer,
                            units=par.num_hidden,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=par.dropout_rate,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    output_layer = tf.layers.dense(inputs=dropout,
                                   units=par.img_width * par.img_height * 3)

    # Calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.absolute_difference(labels=labels,
                                         predictions=output_layer)

    # Configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=par.learning_rate)

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels)}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(config):

    # Load training and eval data
    train_img, train_seg, test_img, test_seg = read_images()

    # Create the Estimator
    segmentation_network = tf.estimator.Estimator(model_fn=model_fn,
                                                  model_dir=par.model_dir)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, at_end=True)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_img},
        y=train_seg,
        batch_size=par.batch_size,
        num_epochs=par.num_epochs_train,
        shuffle=True)

    segmentation_network.train(input_fn=train_input_fn,
                               steps=par.steps,
                               hooks=[logging_hook])
"""
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_img},
        y=test_seg,
        num_epochs=par.num_epochs_eval,
        shuffle=False)

    print(segmentation_network.evaluate(input_fn=eval_input_fn))
"""
if __name__ == "__main__":
    tf.app.run()

