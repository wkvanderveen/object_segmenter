import os
import math
from shutil import rmtree

overwrite_existing_model = True
predict = True
model_dir = './model_data/'
img_width = 32
img_height = 32
train_percentage = 80
max_img = 64
batch_size = 5
num_epochs_train = 100
num_epochs_eval = 5
steps = 300
learning_rate = 0.01
num_hidden = 64
num_filters = 8
filter_size = [5, 5]
dropout_rate = 0.4
seg_threshold = 0.5


plot_layers = {
    "Any":      False,
    "Conv1":    True,
    "Dilated":  True,
    "Conv2":    True,
    "Deconv":   True
}


def rem_existing_model():
    """Delete map containing model metadata."""
    if overwrite_existing_model and os.path.exists(model_dir):
        rmtree(model_dir)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
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
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)
