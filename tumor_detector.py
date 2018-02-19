from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config as cf
import scipy.signal as spsig
import utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress warnings
tf.logging.set_verbosity(tf.logging.INFO)   # display tensorflow info


def visualize_conv(filters, layer, mode, input_image=None):
    """Accept 4-D numpy array (width, height, n_channels, n_filters).

    If 'mode' parameter equals 'weights', then this function will plot
    the convolutional filterweights 'n_channels' times.

    If 'mode' parameter equals 'output', then this function will require
    a parameter 'input_image'.  This image will be convolved with the
    filters and the result will be plotted.
    """

    # Initialize target directory for the plots
    plot_dir_path = os.path.join(cf.plot_dir, 'conv{}_{}'.format(layer, mode))
    utils.prepare_dir(plot_dir_path, empty=True)

    # Extract number of channels and filters from 4-D arrays
    num_channels = filters.shape[2]
    num_filters = filters.shape[3]

    # Set width and height of subplot grid, based on number of filters
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # Plot every channel individually
    for ch in range(num_channels):

        # Print progress information
        if cf.verbose:
            print("Plotting the {} of convolution layer {}: {:>3} of {:>3}"
                  .format(mode, layer, ch+1, num_channels))

        # Optionally convolve each filter with a single input image
        if mode == 'output':

            # Isolate channel and reshape to suitable convolution format
            channel = np.reshape(a=filters[:, :, ch, :],
                                 newshape=(32, 5, 5),
                                 order='C')
            convolutions = np.zeros((32, 24, 24))

            # Convolve every filter in this channel
            for i in range(len(channel)):
                convolutions[i] = spsig.convolve2d(in1=channel[i],
                                                   in2=input_image,
                                                   mode="valid")

            # Normalize plot by storing minimum and maximum values
            v_min = np.min(convolutions)
            v_max = np.max(convolutions)

        # If mode is not 'output', then find the normalization values
        # for the filter weights
        else:
            v_min = np.min(filters)
            v_max = np.max(filters)

        # Instantiate figure for this channel
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))

        # Iterate over subplots and plot image of filter or convolution
        for l, ax in enumerate(axes.flat):

            if mode == 'output':
                img = convolutions[l]
                ax.imshow(img,
                          vmin=v_min,
                          vmax=v_max,
                          interpolation='nearest',
                          cmap='binary')

            else:
                img = filters[:, :, ch, l]
                ax.imshow(img,
                          vmin=v_min,
                          vmax=v_max,
                          interpolation='nearest',
                          cmap='seismic')

            # Remove axis labels
            ax.set_xticks([])
            ax.set_yticks([])

        # Save this channel's figure in designated directory
        plt.savefig(os.path.join(plot_dir_path, 'channel{}.png'.format(ch)),
                    bbox_inches='tight')
        plt.close('all')

    print("Plotting complete! ",
          "These plots can be found in directory {}.\n".format(plot_dir_path))


def model_fn(features, labels, mode):
    """Model function."""

    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional layer #1
    conv1 = tf.layers.conv2d(name="Conv1",
                             inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)

    # Convolutional layer #2 and pooling layer #2
    conv2 = tf.layers.conv2d(name="Conv2",
                             inputs=pool1,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)

    # Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=cf.num_hidden,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=cf.dropout_rate,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout,
                             units=10)

    # Generate predictions (for PREDICT and EVAL mode)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph.  It is used for PREDICT and by the
        # 'logging_hook'.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=cf.learning_rate)

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(config):

    # Optionally overwrite existing metadata by removing its folder
    if cf.overwrite_existing_model:
        if (cf.verbose):
            print("\nRemoving old model in {}...\n".format(cf.model_dir))
        cf.rem_existing_model()

    if (cf.verbose):
        print("\nLoading data...\n")

    # Load training and eval data
    training_data = read_images(train_dir)
    eval_data = read_images(eval_dir)

    if (cf.verbose):
        print("\nCreating Estimator...\n")

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                              model_dir=cf.model_dir)

    if (cf.verbose):
        print("\nSet up logging for predictions...\n")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, at_end=True)

    if (cf.verbose):
        print("\nTraining model...\n")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=cf.batch_size,
        num_epochs=cf.num_epochs_train,
        shuffle=True)

    segmentation_network.train(input_fn=train_input_fn,
                           steps=cf.steps,
                           hooks=[logging_hook])

    if (cf.verbose):
        print("\nEvaluating model...\n")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=cf.num_epochs_eval,
        shuffle=False)

    print(segmentation_network.evaluate(input_fn=eval_input_fn))

    # Load convolution layer values for optional visualization plotting
    conv1 = segmentation_network.get_variable_value("Conv1/kernel")
    conv2 = segmentation_network.get_variable_value("Conv2/kernel")

    # Load single test image for optional predictions
    single_image = eval_data[1]

    # Optionally plot the convolution layer filter weights
    if cf.plot_conv_weights:
        visualize_conv(filters=conv1,
                       layer=1,
                       mode='weights')

        visualize_conv(filters=conv2,
                       layer=2,
                       mode='weights')

    # Optionally convolve a single test image with convolution filter
    # weights to illustrate filters
    if cf.plot_conv_output:
        single_image_reshaped = np.reshape(single_image, (28, 28))
        visualize_conv(filters=conv1,
                       layer=1,
                       mode='output',
                       input_image=single_image_reshaped)

        visualize_conv(filters=conv2,
                       layer=2,
                       mode='output',
                       input_image=single_image_reshaped)

    # Optionally classify an individual test image
    if cf.predict_afterwards:
        if cf.verbose:
            print("\nPredicting one test instance...\n")

        plot_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": single_image},
            batch_size=1,
            shuffle=False)

        for single_predict in mnist_classifier.predict(plot_input_fn):
            print(single_predict)
            print("Predicted class = {}".format(single_predict['classes']))
            print("Probablity = {}\n".format(single_predict['probabilities']))


if __name__ == "__main__":
    tf.app.run()
