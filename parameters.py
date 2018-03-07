"""Parameters and helper functions for residual segmentation network.

   Copyright 2018 Werner van der Veen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os
import math
import errno
from shutil import rmtree

# Information & diagnostics parameters
predict = True
save_predictions = True  # 'True' in hyperparameter optimization
plot_filters = True  # 'False' in hyperparameter optimization
overwrite_existing_model = True  # 'True' in hyperparameter optimization
overwrite_existing_plot = True
plot_layers = {
    "downward":     True,
    "upward":       True,
    "dilated_conv": True,
    "upconv":       True
}

# Network parameters
layer_depth = 2
block_depth = 2  # minimally 2
num_hidden = 32
num_filters = 3
filter_size = [4, 4]
dropout_rate = 0.4
optimizer = "Adadelta"

# Input parameters
img_width = 64
img_height = 64
max_img = 400
batch_size = 10
train_percentage = 90

# Other parameters
model_dir = './model_data/'
plot_dir = './conv_plots/'
pred_dir = './predictions/'
steps = 10
num_epochs_train = 5
num_epochs_eval = 5
learning_rate = 0.01

# Hyperparameter optimizer parameters
hyperparameter1_search = {
    "Name": "num_hidden",  # choose from 'network parameters'
    "min_val": 50,
    "max_val": 70,  # exclusive, must be larger than min_val
    "step": 10
}

hyperparameter2_search = {
    "Name": "num_filters",  # choose from 'network parameters'
    "min_val": 6,
    "max_val": 7,  # exclusive, must be larger than min_val
    "step": 1
}


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
