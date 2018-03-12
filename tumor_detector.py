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
import parameters as par
import utils
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # suppress/display warnings
tf.logging.set_verbosity(tf.logging.INFO)   # display tensorflow info


def get_data():
    """Read data from file and return as train and test sets containing
    sampled input and label images.
    """

    img, seg = utils.read_images()

    train_img, train_seg, test_img, test_seg = utils.sample_images(img, seg)

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
    shape_layers = []  # 1-D: block indices
    downward_conv_layers = []  # 2-D: block and layer indices
    down_batch_norm = []  # 2-D: block and layer indices
    downward_dense_layers = []  # 2-D: block and layer indices
    down_dropout = []  # 2-D: block and layer indices
    upconv = []  # 1-D: block indices
    upward_conv_layers = []  # 2-D: block and layer indices
    up_batch_norm = []  # 2-D: block and layer indices
    upward_dense_layers = []  # 2-D: block and layer indices
    up_dropout = []  # 2-D: block and layer indices

    # Build the "downward" blocks that dilate and convolute repeatedly.
    for block_idx in range(par.block_depth):
        # Initialize this block's "layer" dimension to store tensors.
        downward_conv_layers.append([])
        down_batch_norm.append([])
        downward_dense_layers.append([])
        down_dropout.append([])

        # If this is not the first downward block, read the previous
        # dilation layer.
        # Else, read the input layer.
        if block_idx is not 0:
            shape_layers.append(tf.layers.conv2d(
                inputs=down_dropout[block_idx-1][par.layer_depth-1],
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
                inputs=down_dropout[block_idx][layer_idx-1]
                if layer_idx > 0 else shape_layers[block_idx],
                name=f"downw_convo_block_{block_idx}_layer_{layer_idx}",
                filters=par.num_filters,
                kernel_size=par.filter_size,
                padding="same",
                bias_initializer=tf.random_normal_initializer))

            # Batch Normalization.
            down_batch_norm[block_idx].append(tf.layers.batch_normalization(
                inputs=downward_conv_layers[block_idx][layer_idx],
                name=f"downw_batch_norm_{block_idx}_{layer_idx}"))

            # Dense layer.
            downward_dense_layers[block_idx].append(tf.layers.dense(
                inputs=down_batch_norm[block_idx][layer_idx],
                name=f"downw_dense_block_{block_idx}_layer_{layer_idx}",
                units=par.num_hidden,
                activation=tf.nn.leaky_relu,
                bias_initializer=tf.random_normal_initializer))

            # Dropout.
            down_dropout[block_idx].append(tf.layers.dropout(
                inputs=downward_dense_layers[block_idx][layer_idx],
                rate=par.dropout_rate,
                name=f"downw_dropout_block_{block_idx}_layer_{layer_idx}"))

    # Build the "upward" blocks that use upconvolution and
    # convolution-dense blocks. Deepest block is classified "downward",
    # so there is one less "upward" block.
    for block_idx in range(par.block_depth-1):
        # Initialize this block's "layer" dimension to store tensors.
        upward_conv_layers.append([])
        up_batch_norm.append([])
        upward_dense_layers.append([])
        up_dropout.append([])

        # Upconvolute from the previous upward block. If this is the
        # first upward block, read from the deepest downward block.
        upconv.append(tf.layers.conv2d_transpose(
            inputs=up_dropout[block_idx-1][par.layer_depth-1]
            if block_idx > 0
            else down_dropout[par.block_depth-1][par.layer_depth-1],
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
                inputs=up_dropout[block_idx][layer_idx-1]
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

            # Batch Normalization.
            up_batch_norm[block_idx].append(tf.layers.batch_normalization(
                inputs=upward_conv_layers[block_idx][layer_idx],
                name=f"up_batch_norm_{block_idx}_{layer_idx}"))

            # Dense layer.
            upward_dense_layers[block_idx].append(tf.layers.dense(
                inputs=up_batch_norm[block_idx][layer_idx],
                name=f"upwrd_dense_block_{block_idx}_layer_{layer_idx}",
                units=par.num_hidden,
                activation=tf.nn.leaky_relu,
                bias_initializer=tf.random_normal_initializer))

            # Dropout.
            up_dropout[block_idx].append(tf.layers.dropout(
                inputs=upward_dense_layers[block_idx][layer_idx],
                rate=par.dropout_rate,
                name=f"up_dropout_block_{block_idx}_layer_{layer_idx}"))

    # Output dense layer
    output_dense = tf.layers.dense(
        inputs=up_dropout[par.block_depth-2][par.layer_depth-1],
        name="dense_output_layer",
        units=1,
        activation=tf.sigmoid,
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
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
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
            utils.prepare_dir(par.model_dir, empty=True)
        if par.save_predictions and par.predict:
            utils.prepare_dir(par.pred_dir, empty=True)
        if par.overwrite_existing_plot:
            utils.prepare_dir(par.plot_dir, empty=True)

    # Start timer
    start_time = time.time()

    # Load training and testing data
    train_img, train_seg, test_img, test_seg = get_data()

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
        utils.predict_image(input_=test_img[:1],
                            label=test_seg[:1],
                            pred_fn=tumor_detector.predict)

    if par.plot_filters:
        for block_idx in range(par.block_depth):
            print(f"\nPlotting convolution filters for block ",
                  f"{block_idx}/{par.block_depth-1}")
            print("\tPlotting dilated convolution filters...")
            if par.plot_layers["dilated_conv"] and block_idx > 0:
                utils.plot_conv(filters=tumor_detector.get_variable_value(
                                  f"shape_layer_block_{block_idx}/kernel"),
                                name=["Dilated Convolution", "dilconv"],
                                block=block_idx)

            print("\tPlotting upconvolution filters...")
            if par.plot_layers["upconv"] and block_idx < par.block_depth-1:
                utils.plot_conv(filters=tumor_detector.get_variable_value(
                                  f"upconv_layer_block_{block_idx}/kernel"),
                                name=["Upconvolution", "upconv"],
                                block=block_idx)

            print("\tPlotting convolution filters...")
            for layer_idx in range(par.layer_depth):
                if par.plot_layers["downward"]:
                    utils.plot_conv(filters=tumor_detector.get_variable_value(
                                      f"downw_convo_block_{block_idx}_" +
                                      f"layer_{layer_idx}/kernel"),
                                    name=["Downward Convolution", "downward"],
                                    block=block_idx,
                                    layer=layer_idx)
                if par.plot_layers["upward"] and block_idx < par.block_depth-1:
                    utils.plot_conv(filters=tumor_detector.get_variable_value(
                                      f"upwrd_convo_block_{block_idx}_" +
                                      f"layer_{layer_idx}/kernel"),
                                    name=["Upward Convolution", "upward"],
                                    block=block_idx,
                                    layer=layer_idx)

    return evaluation['accuracy']


if __name__ == "__main__":
    tf.app.run()
