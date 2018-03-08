"""Residual segmentation network.
   This file should not be modified -- for changing variables, go to
   parameters.py. See README.md file for instructions.

   Copyright 2018 Werner van der Veen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied. See the License for the specific language governing
   permissions and limitations under the License.
"""
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # suppress/display warnings
tf.logging.set_verbosity(tf.logging.INFO)   # display tensorflow info

def predict_image(input_, label, pred_fn):
    """Generate output from an input image and a prediction function."""

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": input_},
        batch_size=1,
        shuffle=False)

    for single_predict in pred_fn(pred_input_fn):
        imgs = [input_[0, :, :], single_predict["output"], label[0, :, :]]

    # Initialize figure
    fig = plt.figure()

    # Add the three images as subplots
    for img in range(3):
        sub = fig.add_subplot(1, 3, img+1)
        sub.set_title(['Input', 'Prediction', 'Label'][img])
        plt.imshow(imgs[img], cmap='gray')
        plt.axis('off')

    if par.save_predictions:
        plt.savefig(os.path.join(
            par.pred_dir, str(len([f for f in os.listdir(par.pred_dir)]))))
        plt.close('all')
    else:
        plt.show()


def plot_conv(filters, name, block, layer=None):
    """Plot the filters of the convolutional layers and save to file."""
    plot_dir_path = os.path.join(
        par.plot_dir,
        f"{name[1]}")
    par.prepare_dir(plot_dir_path, empty=False)

    n_filters = filters.shape[3]

    grid_r, grid_c = par.get_grid_dim(n_filters)

    v_min = np.min(filters)
    v_max = np.max(filters)

    # Instantiate figure for this channel
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    fig.suptitle(f"{name[0]} filters for block {block}" +
                 ("" if layer is None else f" and layer {layer}"))
    fig.text(0, 0, (f"number of filters: {n_filters}\nfilter size: ",
                    f"{filters.shape[0]}x{filters.shape[1]}"))

    # Iterate over subplots and plot image of filter or convolution
    for filt, axis in enumerate(axes.flat):

        img = filters[:, :, 0, filt]
        axis.imshow(img,
                    vmin=v_min,
                    vmax=v_max,
                    interpolation='nearest',
                    cmap='seismic')

        # Remove axis labels
        axis.set_xticks([])
        axis.set_yticks([])

    plt.savefig(os.path.join(
        plot_dir_path,
        f"{name[1]}_b{block}" + ("" if layer is None else f"_l{layer}")))
    plt.close('all')


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

        print(f"Reading images... Now at {100*n_files/par.max_img:.2f}%",
              end='\r',
              flush=True)

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
        print((f"Sampling images... Now at " +
               f"{100*(c_train+c_test+1)/n_files:.2f}%"),
              end='\r',
              flush=True)

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

    print("\nSampling images completed!\n")

    return train_img, train_seg, test_img, test_seg


def model_fn(features, labels, mode):
    """Model function of the segmentation network.
    The network is an encoder-decoder network with residual connections.
    """

    # Input layer: reshape the data to a variable batch of 2-D images.
    input_layer = tf.reshape(tensor=features["x"],
                             shape=[-1, par.img_width, par.img_height, 1],
                             name="shape_layer_block_0")

    # Initialize the main layers of the network.
    # The first half of the network consists out of N dilated
    # convolutions, which are alternated with blocks of M combinations
    # of same-size convolution and dense layers.
    # The second half of the network consists out of N upconvolutions
    # and residual connections to corresponding downward layers, again
    # alternated with M convolution-dense combinations.
    upward_conv_layers = []  # 2-D: blocks and layer indices
    upward_dense_layers = []  # 2-D: blocks and layer indices
    downward_conv_layers = []  # 2-D: blocks and layer indices
    downward_dense_layers = []  # 2-D: blocks and layer indices
    shape_layers = []  # 1-D: blocks
    upconv = []  # 1-D: blocks

    # Build the "downward" blocks that dilate and convolute repeatedly.
    for block_idx in range(par.block_depth):
        # Initialize this block's "layer" dimension to store tensors.
        downward_dense_layers.append([])
        downward_conv_layers.append([])

        # If this is not the first downward block, read the previous
        # dilation layer.
        # Else, read the input layer.
        if block_idx is not 0:
            shape_layers.append(tf.layers.conv2d(
                inputs=downward_dense_layers[block_idx-1][par.layer_depth-1],
                name=f"shape_layer_block_{block_idx}",
                filters=par.num_filters,
                kernel_size=par.filter_size,
                padding="same",
                dilation_rate=(2, 2),
                bias_initializer=tf.random_normal_initializer))
        else:
            shape_layers.append(input_layer)

        # Construct the convolution-dense layers of this downward block.
        for layer_idx in range(par.layer_depth):
            # Convolution layer. Read from previous layer if available.
            # Else, read from previous block.
            downward_conv_layers[block_idx].append(tf.layers.conv2d(
                inputs=downward_dense_layers[block_idx][layer_idx-1]
                if layer_idx > 0 else shape_layers[block_idx],
                name=f"downw_convo_block_{block_idx}_layer_{layer_idx}",
                filters=par.num_filters,
                kernel_size=par.filter_size,
                padding="same",
                bias_initializer=tf.random_normal_initializer))

            # Dense layer.
            downward_dense_layers[block_idx].append(tf.layers.dense(
                inputs=downward_conv_layers[block_idx][layer_idx],
                name=f"downw_dense_block_{block_idx}_layer_{layer_idx}",
                units=par.num_hidden,
                activation=tf.nn.relu,
                bias_initializer=tf.random_normal_initializer))

    # Build the "upward" blocks that use upconvolution and
    # convolution-dense blocks. Deepest block is classified "downward",
    # so there is one less "upward" block.
    for block_idx in range(par.block_depth-1):
        # Initialize this block's "layer" dimension to store tensors.
        upward_dense_layers.append([])
        upward_conv_layers.append([])

        # Upconvolute from the previous upward block. If this is the
        # first upward block, read from the deepest downward block.
        upconv.append(tf.layers.conv2d_transpose(
            inputs=upward_dense_layers[block_idx-1][par.layer_depth-1]
            if block_idx > 0
            else downward_dense_layers[par.block_depth-1][par.layer_depth-1],
            name=f"upconv_layer_block_{block_idx}",
            filters=par.num_filters,
            kernel_size=par.filter_size,
            padding="same"))

        # Construct the convolution-dense layers of this upward block.
        for layer_idx in range(par.layer_depth):

            # Convolution layer. Read from previous layer if available.
            # Else, read from previous block and the residual connection
            # from the corresponding downward block.
            upward_conv_layers[block_idx].append(tf.layers.conv2d(
                inputs=upward_dense_layers[block_idx][layer_idx-1]
                if layer_idx > 0
                else tf.concat(
                    [upconv[block_idx],
                     shape_layers[par.block_depth-2-block_idx]],
                    -1),
                name=f"upwrd_convo_block_{block_idx}_layer_{layer_idx}",
                filters=par.num_filters,
                kernel_size=par.filter_size,
                padding="same",
                bias_initializer=tf.random_normal_initializer))

            # Dense layer.
            upward_dense_layers[block_idx].append(tf.layers.dense(
                inputs=upward_conv_layers[block_idx][layer_idx],
                name=f"upwrd_dense_block_{block_idx}_layer_{layer_idx}",
                units=par.num_hidden,
                activation=tf.nn.relu,
                bias_initializer=tf.random_normal_initializer))

    # Output dense layer
    output_dense = tf.layers.dense(
        inputs=upward_dense_layers[par.block_depth-2][par.layer_depth-1],
        name="dense_output_layer",
        units=1,
        activation=tf.nn.relu,
        bias_initializer=tf.random_normal_initializer)

    # Final output layer (batch of 2-D images)
    output = tf.reshape(tensor=output_dense,
                        shape=[-1, par.img_width, par.img_height],
                        name="final_output_layer")

    # Save the output (for PREDICT mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={"output": output})

    # Calculate loss
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=output)

    # Configure the training op (for TRAIN mode)
    optimizers = {
        "GradientDescent":  tf.train.GradientDescentOptimizer,
        "Adadelta":         tf.train.AdadeltaOptimizer
    }

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = optimizers[par.optimizer](
            learning_rate=par.learning_rate)

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step(),
                                      name="optimizer")

        # Set up logging
        logging_hook = tf.train.LoggingTensorHook(
            tensors={"step": "optimizer"}, every_n_iter=par.step_log_interval)

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=output)
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(config):
    # If config contains additional parameters, we are doing grid search
    # for hyperparameters. In this case, overwrite relevant parameters
    # from 'par' module, and reduce verbosity.
    if len(config) > 1:
        tf.logging.set_verbosity(tf.logging.WARN)
        par.save_predictions = True
        par.plot_filters = False
        for [par_name, par_value] in config[1:]:
            if par_name is "num_hidden":
                par.num_hidden = par_value
            if par_name is "num_filters":
                par.num_filters = par_value
            if par_name is "layer_depth":
                par.layer_depth = par_value
            if par_name is "block_depth":
                par.block_depth = par_value
            if par_name is "filter_size":
                par.filter_size = [par_value, par_value]
            if par_name is "dropout_rate":
                par.dropout_rate = par_value
            if par_name is "optimizer":
                par.optimizer = par_value

    else:
        # Standard case. Optionally overwrite existing model by emptying
        # its directory
        if par.overwrite_existing_model:
            par.prepare_dir(par.model_dir, empty=True)
        if par.save_predictions and par.predict:
            par.prepare_dir(par.pred_dir, empty=True)
        if par.overwrite_existing_plot:
            par.prepare_dir(par.plot_dir, empty=True)

    # Start timer
    start_time = time.time()

    # Load training and testing data
    train_img, train_seg, test_img, test_seg = read_images()

    # Create the Estimator
    print("Creating Estimator...")
    tumor_detector = tf.estimator.Estimator(model_fn=model_fn,
                                            model_dir=par.model_dir)
    print("Creating Estimator completed!\n")

    # Train the model
    print("Training model...")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_img},
        y=train_seg,
        batch_size=par.batch_size,
        num_epochs=par.num_epochs_train,
        shuffle=True)

    tumor_detector.train(input_fn=train_input_fn,
                         steps=par.steps)
    print("Training model completed!\n")

    # Evaluate the model and print results
    print("Evaluating model...")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_img},
        y=test_seg,
        num_epochs=par.num_epochs_eval,
        shuffle=True)

    evaluation = tumor_detector.evaluate(input_fn=eval_input_fn)
    print("Evaluating model completed!\n")

    # Print time elapsed
    print(time.strftime(
        "Time elapsed: %H:%M:%S", time.gmtime(int(time.time() - start_time))))

    # Optionally predict a random test image
    if par.predict:
        predict_image(input_=test_img[:1],
                      label=test_seg[:1],
                      pred_fn=tumor_detector.predict)

    if par.plot_filters:
        for block_idx in range(par.block_depth):
            print(f"\nPlotting convolution filters for block ",
                  f"{block_idx}/{par.block_depth-1}")
            print("\tPlotting dilated convolution filters...")
            if par.plot_layers["dilated_conv"] and block_idx > 0:
                plot_conv(filters=tumor_detector.get_variable_value(
                                  f"shape_layer_block_{block_idx}/kernel"),
                          name=["Dilated Convolution", "dilconv"],
                          block=block_idx)

            print("\tPlotting upconvolution filters...")
            if par.plot_layers["upconv"] and block_idx < par.block_depth-1:
                plot_conv(filters=tumor_detector.get_variable_value(
                                  f"upconv_layer_block_{block_idx}/kernel"),
                          name=["Upconvolution", "upconv"],
                          block=block_idx)

            print("\tPlotting convolution filters...")
            for layer_idx in range(par.layer_depth):
                if par.plot_layers["downward"]:
                    plot_conv(filters=tumor_detector.get_variable_value(
                                      f"downw_convo_block_{block_idx}_" +
                                      f"layer_{layer_idx}/kernel"),
                              name=["Downward Convolution", "downward"],
                              block=block_idx,
                              layer=layer_idx)
                if par.plot_layers["upward"] and block_idx < par.block_depth-1:
                    plot_conv(filters=tumor_detector.get_variable_value(
                                      f"upwrd_convo_block_{block_idx}_" +
                                      f"layer_{layer_idx}/kernel"),
                              name=["Upward Convolution", "upward"],
                              block=block_idx,
                              layer=layer_idx)

    return evaluation['accuracy']


if __name__ == "__main__":
    tf.app.run()
