import os
from shutil import rmtree

overwrite_existing_model = True
predict = True
model_dir = './model_data/'
img_width = 150
img_height = 150
img_channels = 3
seg_channels = 1
train_percentage = 80
max_img = 50
batch_size = 1
num_epochs_train = 100
num_epochs_eval = 5
steps = 100
learning_rate = 0.01
num_hidden = 128
dropout_rate = 0.1
seg_threshold = 0.5


def rem_existing_model():
    """Delete map containing model metadata."""
    if overwrite_existing_model and os.path.exists(model_dir):
        rmtree(model_dir)
