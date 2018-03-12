"""Helper functions for residual segmentation network.
   This file should not be modified -- for changing variables, go to
   parameters.py.
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

import os
import cv2
import numpy as np
import parameters as par
import random as rand
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import errno
from sklearn.preprocessing import normalize
from shutil import rmtree


def read_images():

    # Set directories containing the images and segmentation objects.
    # Initialize the lists where the input and segmentation images will
    # be stored.
    img_list = []
    seg_list = []

    n_files = 0

    # Append the input and segmentation images to the respective lists.
    for seg_file in os.listdir(par.seg_dir):
        filename = seg_file.split('.')[0]

        # Stop adding more images if the image limit is reached.
        if n_files > par.max_img:
            break

        # If the input image that corresponds to this segmentation image
        # cannot be located, then skip this segmentation image.
        if not os.path.isfile(par.img_dir + filename + '.jpg'):
            continue

        print(f"Reading images... Now at {100*n_files/par.max_img:.2f}%",
              end='\r',
              flush=True)

        n_files += 1

        # Read the images from their files.
        img = cv2.imread(par.img_dir + filename + '.jpg')

        seg = cv2.imread(par.seg_dir + seg_file)

        # Resize the images to the standard size (from the parameters)
        img = cv2.resize(
            src=img,
            dsize=(par.img_width, par.img_height),
            interpolation=cv2.INTER_NEAREST)
        seg = cv2.resize(
            src=seg,
            dsize=(par.img_width, par.img_height),
            interpolation=cv2.INTER_NEAREST)

        # Convert BGR to greyscale color format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(float)

        # Convert segmentation image to binary color format.
        seg = np.ceil(np.amax(seg, axis=2))

        # Normalize the image values.
        img = (img - np.min(img)) / (np.ptp(img)/2) - 1  # between [-1,1]
        seg = (seg - np.min(seg)) / (np.ptp(seg))  # between [0,1]

        # Append resized images to their respective lists.
        img_list.append(img)
        seg_list.append(seg)

    print("\nReading images completed!\n")

    return img_list, seg_list


def sample_images(img, seg):
    # Initialize numpy arrays
    n_files = len(img)
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
    prepare_dir(plot_dir_path, empty=False)

    n_filters = filters.shape[3]

    grid_r, grid_c = get_grid_dim(n_filters)

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


def rem_dir(directory):
    """Delete a directory."""
    if os.path.exists(directory):
        rmtree(directory)


def empty_dir(path):
    """Delete all files and folders in a directory."""
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
        except Exception as e:
            print('Warning: {}'.format(e))


def get_grid_dim(x):
    """Transform x into product of two integers."""
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """Compute the factors of a positive integer."""
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def create_dir(path):
    """Create a directory."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)


def print_big_title(text):
    """Pretty print a large title that stands out in the terminal."""
    size = 80
    sym = '*'
    print(f"\n\n{sym*size}")
    for line in text:
        print(f"***** {line}{' '*(size-12-len(line))} *****")
    print(f"{sym*size}\n\n")
