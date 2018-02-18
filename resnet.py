# Copyright 2018 W.K. van der Veen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Contains definitions for the preactivation form of Residual Networks
(also known as ResNet v2).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


########################################################################
# Functions for input processing.
########################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1,
                           num_parallel_calls=1):
    """ Given a data set with raw records, parse each record into images
    and labels, and return an iterator over the records.

    Args:
        dataset: A data set representing raw records.
        is_training: A boolean denoting whether the input is for
            training.
        batch_size: The number of samples per batch.
        shuffle_buffer: The buffer size to use when shuffling records.
            A larger value results in better randomness, but smaller
            values reduce startup time and use less memory.
        parse_record_fn: A function that takes a raw record and returns
            the corresponding (image, label) pair.
        num_epochs: The number of epochs to repeat the dataset.
        num_parallel_calls: The number of records that are processed in
            parallel. This can be optimized per data set but for
            generally homogeneous data sets, should be approximately the
            number of available GPU cores.

    Returns:
        Data set of (image, label) pairs ready for iteration.
    """

    # We prefetch a batch at a time. This can help smooth out the time
    # taken to load input files as we go through shuffling and
    # processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffle the records. Note that we shuffle before repeating to
        # ensure that the shuffling respects epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # If we are training over multiple epochs before evaluating, repeat
    # the data set for the appropriate number of epochs.
    dataset = dataset.repeat(num_epochs)

    # Parse the raw records into images and labels.
    dataset = dataset.map(lambda value: parse_record_fn(value, is_training),
                          num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(batch_size)

    # Operations between the final prefetch and the get_next call to the
    # iterator will happen synchronously during run time. We prefetch
    # here again to background all of the above processing work and keep
    # it out of the critical training path.
    dataset = dataset.prefetch(1)

    return dataset


########################################################################
# Functions building the ResNet model.
########################################################################
def batch_norm_relu(inputs, training, data_format):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):

    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def building_block(inputs, filters, training, projection_shortcut, strides,
    data_format):

    shortcut = inputs
    inputs = batch_norm_relu(inputs, training, data_format)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut


def bottleneck_block(inputs, filters, training, projection_shortcut, strides,
    data_format):

    shortcut = inputs
    inputs = batch_norm_relu(inputs, training, data_format)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, training, name,
    data_format):

    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


class Model(object):

    def __init__(self, resnet_size, num_classes, num_filters, kernel_size,
                 conv_stride, first_pool_size, first_pool_stride,
                 second_pool_size, second_pool_stride, block_fn, block_sizes,
                 block_strides, final_size, data_format=None):

        self.resnet_size = resnet_size

        if not data_format:
            data_format = ('channels_first' if tf.test.is_built_with_cuda()
                           else 'channels_last')

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        self.block_fn = block_fn
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size

    def __call__(self, inputs, training):

        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = conv2d_fixed_padding(inputs=inputs,
                                          filters=self.num_filters,
                                          kernel_size=self.kernel_size,
                                          strides=self.conv_stride,
                                          data_format=self.data_format)

            inputs = tf.identity(inputs, 'initial_conv')

        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=self.first_pool_size,
                strides=self.first_pool_stride, padding='SAME',
                data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2**i)
            inputs = block_layer(
                inputs=inputs, filters=num_filters, block_fn=self.block_fn,
                blocks=num_blocks, strides=self.block_strides[i],
                training=training, name='block_layer{}'.format(i + 1),
                data_format=self.data_format)

        inputs = batch_norm_relu(inputs, training, self.data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=self.second_pool_size,
            strides=self.second_pool_stride, padding='VALID',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')

        inputs = tf.reshape(inputs, [-1, self.final_size])
        inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs


def learning_rate_with_decay(batch_size, batch_denom, num_images,
                             boundary_epochs, decay_rates):

    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, loss_filter_fn=None):

    tf.summary.image('images', features, max_outputs=6)

    model = model_class(resnet_size, data_format)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    if not loss_filter_fn:
        def loss_filter_fn(name):
            return 'batch_normalization' not in name

    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if loss_filter_fn(v.name)])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics)


def resnet_main(flags, model_function, input_function):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags.model_dir, config=run_config,
        params={
            'resnet_size': flags.resnet_size,
            'data_format': flags.data_format,
            'batch_size': flags.batch_size,
        })

    for _ in range(flags.train_epochs // flags.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

    print('Starting a training cycle.')

    def input_fn_train():
        return input_function(True, flags.data_dir, flags.batch_size,
                              flags.epochs_per_eval, flags.num_parallel_calls)

    classifier.train(input_fn=input_fn_train, hooks=[logging_hook])

    print('Starting to evaluate.')

    def input_fn_eval():
        return input_function(False, flags.data_dir, flags.batch_size,
                              1, flags.num_parallel_calls)

    eval_results = classifier.evaluate(input_fn=input_fn_eval)
    print(eval_results)


class ResnetArgParser(argparse.ArgumentParser):

    def __init__(self, resnet_size_choices=None):
        super(ResnetArgParser, self).__init__()
        self.add_argument(
            '--data_dir', type=str, default='/tmp/resnet_data',
            help='The directory where the input data is stored.')

        self.add_argument(
            '--num_parallel_calls', type=int, default=5,
            help='The number of records that are processed in parallel '
            'during input processing. This can be optimized per data set but '
            'for generally homogeneous data sets, should be approximately the '
            'number of available CPU cores.')

        self.add_argument(
            '--model_dir', type=str, default='/tmp/resnet_model',
            help='The directory where the model will be stored.')

        self.add_argument(
            '--resnet_size', type=int, default=50,
            choices=resnet_size_choices,
            help='The size of the ResNet model to use.')

        self.add_argument(
            '--train_epochs', type=int, default=100,
            help='The number of epochs to use for training.')

        self.add_argument(
            '--epochs_per_eval', type=int, default=1,
            help='The number of training epochs to run between evaluations.')

        self.add_argument(
            '--batch_size', type=int, default=32,
            help='Batch size for training and evaluation.')

        self.add_argument(
            '--data_format', type=str, default=None,
            choices=['channels_first', 'channels_last'],
            help='A flag to override the data format used in the model. '
                 'channels_first provides a performance boost on GPU but '
                 'is not always compatible with CPU. If left unspecified, '
                 'the data format will be chosen automatically based on '
                 'whether TensorFlow was built for CPU or GPU.')
