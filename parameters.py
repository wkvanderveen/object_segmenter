import os
from shutil import rmtree

overwrite_existing_model = True
model_dir = './model_data/'
img_width = 64
img_height = 64
img_channels = 3
train_percentage = 80
max_img = 1000
batch_size = 50
num_epochs_train = 500
num_epochs_eval = 10
steps = 200
learning_rate = 0.1
num_hidden = 200
dropout_rate = 0.1


def rem_existing_model():
    """Delete map containing model metadata."""
    if overwrite_existing_model and os.path.exists(model_dir):
        rmtree(model_dir)
