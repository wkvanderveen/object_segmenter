from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import parameters as par
import matplotlib.pyplot as plt
import cv2
import random as rand
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress warnings
tf.logging.set_verbosity(tf.logging.INFO)   # display tensorflow info


def display_image(img):
    plt.imshow(img)
    plt.show()


def read_images():
    seg_dir = './VOC2012-objects/SegmentationObject/'
    img_dir = './VOC2012-objects/JPEGImages/'

    n_files = 0
    seg = []
    img = []

    for seg_file in os.listdir(seg_dir):
        filename = seg_file.split('.')[0]

        if n_files > par.max_img:
            break
        if not os.path.isfile(img_dir + filename + '.jpg'):
            continue

        n_files += 1
        print(f"Now reading file {img_dir}{filename}...")

        resized_seg = cv2.resize(
            src=cv2.imread(seg_dir + seg_file),
            dsize=(par.img_width, par.img_height))
        resized_img = cv2.resize(
            src=cv2.imread(img_dir + filename + '.jpg'),
            dsize=(par.img_width, par.img_height))

        resized_seg = np.asarray(resized_seg)
        resized_img = np.asarray(resized_img)

        resized_seg = resized_seg.astype(float) / 255
        resized_img = resized_img.astype(float) / 255

        seg.append(resized_seg)
        img.append(resized_img)

    n_train = int(par.train_percentage / 100 * n_files)
    n_test = n_files - n_train

    train_img = np.zeros(shape=(n_train,
                                par.img_width,
                                par.img_height,
                                par.img_channels))
    train_seg = np.zeros(shape=(n_train,
                                par.img_width,
                                par.img_height,
                                par.img_channels))
    test_img = np.zeros(shape=(n_test,
                               par.img_width,
                               par.img_height,
                               par.img_channels))
    test_seg = np.zeros(shape=(n_test,
                               par.img_width,
                               par.img_height,
                               par.img_channels))

    count_train = 0
    count_test = 0

    for rand_idx in rand.sample(range(n_files), n_files):
        if count_train < n_train:
            train_img[count_train] = img[rand_idx]
            train_seg[count_train] = seg[rand_idx]
            count_train += 1
        else:
            test_img[count_test] = img[rand_idx]
            test_seg[count_test] = seg[rand_idx]
            count_test += 1

    return train_img, train_seg, test_img, test_seg


def model_fn(features, labels, mode):
    """Model function."""

    # Input layer
    input_layer = tf.reshape(
        features["x"], [-1, par.img_width * par.img_height * par.img_channels])

    dense = tf.layers.dense(inputs=input_layer,
                            units=par.num_hidden,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=par.dropout_rate,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    output_layer = tf.layers.dense(
        inputs=dropout,
        units=par.img_width * par.img_height * par.img_channels,
        activation=tf.nn.relu)

    output = tf.reshape(output_layer,
                        [-1, par.img_width, par.img_height, par.img_channels])

    predictions = {
        "seg_out": output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.absolute_difference(labels=labels,
                                         predictions=output)

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
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["seg_out"])}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(config):

    # Optionally overwrite existing metadata by removing its folder
    if par.overwrite_existing_model:
        par.rem_existing_model()

    # Load training and eval data
    train_img, train_seg, test_img, test_seg = read_images()

    # Create the Estimator
    tumor_detector = tf.estimator.Estimator(model_fn=model_fn,
                                            model_dir=par.model_dir)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_img},
        y=train_seg,
        batch_size=par.batch_size,
        num_epochs=par.num_epochs_train,
        shuffle=True)

    tumor_detector.train(input_fn=train_input_fn,
                         steps=par.steps)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_img},
        y=test_seg,
        num_epochs=par.num_epochs_eval,
        shuffle=False)

    print(tumor_detector.evaluate(input_fn=eval_input_fn))

    single_image = test_img[:1]
    display_image(single_image[0, :, :, :])

    plot_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": single_image},
        batch_size=1,
        shuffle=False)

    for single_predict in tumor_detector.predict(plot_input_fn):
        display_image(single_predict["seg_out"])


if __name__ == "__main__":
    tf.app.run()
