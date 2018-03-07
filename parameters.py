"""Parameters and helper functions for residual segmentation network."""

import os
import math
import errno
from shutil import rmtree

predict = True
model_dir = './model_data/'
overwrite_existing_model = True
plot_dir = './conv_plots/'
plot_filters = False
overwrite_existing_plot = True
img_width = 64
img_height = 64
train_percentage = 90
max_img = 400
batch_size = 10
num_epochs_train = 20
num_epochs_eval = 5
steps = 100
learning_rate = 0.01
num_hidden = 56
layer_depth = 1
block_depth = 2 # minimally 2
num_filters = 4
filter_size = [5, 5]
dropout_rate = 0.4
seg_threshold = 0.5
optimizer = "Adadelta"

plot_layers = {
    "downward":     True,
    "upward":       True,
    "dilated_conv": True,
    "upconv":       True
}


def rem_existing_model():
    """Delete map containing model metadata."""
    if overwrite_existing_model and os.path.exists(model_dir):
        rmtree(model_dir)
    if overwrite_existing_plot and os.path.exists(plot_dir):
        rmtree(plot_dir)


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
