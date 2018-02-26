import os
from shutil import rmtree

overwrite_existing_model = True
predict = True
model_dir = './model_data/'
img_width = 128
img_height = 128
img_channels = 3
seg_channels = 1
train_percentage = 80
max_img = 200
batch_size = 20
num_epochs_train = 50
num_epochs_eval = 5
steps = 20
learning_rate = 0.1
num_hidden = 100
dropout_rate = 0.1
seg_threshold = 0.5


def rem_existing_model():
    """Delete map containing model metadata."""
    if overwrite_existing_model and os.path.exists(model_dir):
        rmtree(model_dir)
