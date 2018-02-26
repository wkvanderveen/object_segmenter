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
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress warnings
tf.logging.set_verbosity(tf.logging.INFO)   # display tensorflow info


def plot_conv(filters):
    n_filters = filters.shape[3]

    grid_r, grid_c = par.get_grid_dim(n_filters)

    v_min = np.min(filters)
    v_max = np.max(filters)

    # Instantiate figure for this channel
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    fig.suptitle("Convolutional filters")
    fig.text(0, 0, "number of filters: " + str(n_filters) +
                   "\nfilter size: " + str(filters.shape[0]) +
                   "x" + str(filters.shape[1]))

    # Iterate over subplots and plot image of filter or convolution
    for l, ax in enumerate(axes.flat):

        img = filters[:, :, 0, l]
        ax.imshow(img,
                  vmin=v_min,
                  vmax=v_max,
                  interpolation='nearest',
                  cmap='seismic')

        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def display_prediction(imgs):
    """Display input, prediction, and label images."""

    # Initialize figure
    fig = plt.figure()

    # Add the three images as subplots
    for img in range(3):
        sub = fig.add_subplot(1, 3, img+1)
        sub.set_title(['Input', 'Prediction', 'Label'][img])
        plt.imshow(imgs[img], cmap='gray')
        plt.axis('off')
    plt.show()


def read_images():
    """Read the input-output pairs from folder and return numpy arrays
    of the train and test sets.
    """

    # Set directories containing the images and segmentation objects.
    img_dir = './VOC2012-objects/JPEGImages/'
    seg_dir = './VOC2012-objects/SegmentationObject/'

    # Initialize the lists where the input and segmentation images will
    # be stored.
    img = []
    seg = []

    n_files = 0

    # Append the input and segmentation images to the respective lists.
    for seg_file in os.listdir(seg_dir):
        filename = seg_file.split('.')[0]

        # Stop adding more images if the image limit is reached.
        if n_files > par.max_img:
            break

        # If the input image that corresponds to this segmentation image
        # cannot be located, then skip this segmentation image.
        if not os.path.isfile(img_dir + filename + '.jpg'):
            continue

        #print(f"Reading images... Now at {100*n_files/par.max_img:.2f}%",
        #      end='\r',
        #      flush=True)

        n_files += 1

        # Read the images from their files
        current_img = cv2.imread(img_dir + filename + '.jpg')

        current_seg = cv2.imread(seg_dir + seg_file)

        # Resize the images to the standard size (from the parameters)
        resized_img = cv2.resize(
            src=current_img,
            dsize=(par.img_width, par.img_height),
            interpolation=cv2.INTER_NEAREST)
        resized_seg = cv2.resize(
            src=current_seg,
            dsize=(par.img_width, par.img_height),
            interpolation=cv2.INTER_NEAREST)

        # Convert BGR to greyscale color format
        recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        recolored_img = recolored_img.astype(float)

        # Convert segmentation image to binary color format.
        recolored_seg = np.ceil(np.amax(resized_seg, axis=2))
        recolored_seg = recolored_seg.astype(int)

        # Append resized images to their respective lists.
        img.append(recolored_img)
        seg.append(recolored_seg)

    print("\nReading images completed!\n")

    # Initialize numpy arrays
    n_train = int(par.train_percentage / 100 * n_files)
    n_test = n_files - n_train

    train_img = np.zeros(shape=(n_train,
                                par.img_width,
                                par.img_height),
                         dtype=np.float32)
    train_seg = np.zeros(shape=(n_train,
                                par.img_width,
                                par.img_height),
                         dtype=np.float32)
    test_img = np.zeros(shape=(n_test,
                               par.img_width,
                               par.img_height),
                        dtype=np.float32)
    test_seg = np.zeros(shape=(n_test,
                               par.img_width,
                               par.img_height),
                        dtype=np.float32)

    c_train = 0
    c_test = 0

    # Randomly bifurcate the input and segmentation images into  a train
    # and test set.
    for rand_idx in rand.sample(range(n_files), n_files):
        # print(f"Sampling images... Now at {100*(c_train+c_test)/n_files:.2f}%",
        #       end='\r',
        #       flush=True)

        # Store images in train or test set, depending on whether the
        # limit for the training images is reached.
        if c_train < n_train:
            train_img[c_train] = img[rand_idx] / 256
            train_seg[c_train] = seg[rand_idx] / 256
            c_train += 1
        else:
            test_img[c_test] = img[rand_idx] / 256
            test_seg[c_test] = seg[rand_idx] / 256
            c_test += 1

    return train_img, train_seg, test_img, test_seg


def model_fn(features, labels, mode):
    """Model function of the segmentation network."""

    # Input layer
    input_layer = tf.reshape(
        features["x"], [-1, par.img_width, par.img_height, 1])

    # Down block
    conv1 = tf.layers.conv2d(name="Conv1_layer",
                             inputs=input_layer,
                             filters=par.num_filters,
                             kernel_size=par.filter_size,
                             padding="same",
                             bias_initializer=tf.random_normal_initializer)

    conv_norm1 = tf.layers.batch_normalization(inputs=conv1,
                                               name="Batch_normalization1")

    dense1 = tf.layers.dense(inputs=conv_norm1,
                             units=par.num_hidden,
                             activation=tf.nn.relu,
                             name="Dense1",
                             bias_initializer=tf.random_normal_initializer)

    dil = tf.layers.conv2d(name="Dilation",
                           inputs=dense1,
                           filters=par.num_filters,
                           kernel_size=par.filter_size,
                           padding="same",
                           dilation_rate=(2, 2),
                           bias_initializer=tf.random_normal_initializer)

    #################################################################
    # Up block
    conv2 = tf.layers.conv2d(name="Conv2_layer",
                             inputs=dil,
                             filters=par.num_filters,
                             kernel_size=par.filter_size,
                             padding="same",
                             bias_initializer=tf.random_normal_initializer)

    conv_norm2 = tf.layers.batch_normalization(inputs=conv2,
                                               name="Batch_normalization2")

    dense2 = tf.layers.dense(inputs=conv_norm2,
                             units=par.num_hidden,
                             activation=tf.nn.relu,
                             name="Dense2")

    deconv = tf.layers.conv2d_transpose(inputs=dense2,
                                        filters=par.num_filters,
                                        kernel_size=par.filter_size,
                                        padding="same",
                                        name="Deconv")

    # Output images

    dense3 = tf.layers.dense(inputs=deconv,
                             units=1,
                             activation=tf.nn.relu,
                             name="Dense3",
                             bias_initializer=tf.random_normal_initializer)

    output = tf.reshape(dense3,
                        [-1, par.img_width, par.img_height])

    # Save the output (for PREDICT mode)
    predictions = {"seg_out": output}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels,
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

    # Optionally overwrite existing metadata by removing its directory
    if par.overwrite_existing_model:
        par.rem_existing_model()

    # Start timer
    start_time = time.time()

    # Load training and testing data
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
        shuffle=True)

    print(tumor_detector.evaluate(input_fn=eval_input_fn))

    # Print time elapsed
    print(time.strftime(
        "Time elapsed: %H:%M:%S", time.gmtime(int(time.time() - start_time))))

    # Optionally predict a random test image
    if par.predict:
        single_image = test_img[:1]

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": single_image},
            batch_size=1,
            shuffle=False)

        for single_predict in tumor_detector.predict(pred_input_fn):
            display_prediction([test_img[0, :, :],
                                single_predict["seg_out"],
                                test_seg[0, :, :]])

    if par.plot_layers["Any"]:
        if par.plot_layers["Conv1"]:
            plot_conv(tumor_detector.get_variable_value("Conv1_layer/kernel"))
        if par.plot_layers["Dilated"]:
            plot_conv(tumor_detector.get_variable_value("Dilation/kernel"))
        if par.plot_layers["Conv2"]:
            plot_conv(tumor_detector.get_variable_value("Conv2_layer/kernel"))
        if par.plot_layers["Deconv"]:
            plot_conv(tumor_detector.get_variable_value("Deconv/kernel"))


if __name__ == "__main__":
    tf.app.run()
