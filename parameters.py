"""Parameters for residual segmentation network.

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

# Information & diagnostics parameters
predict = True
save_predictions = False  # 'True' in hyperparameter optimization
overwrite_existing_model = True  # 'True' in hyperparameter optimization
plot_filters = False  # 'False' in hyperparameter optimization
overwrite_existing_plot = True
plot_layers = {
    "downward":     True,
    "upward":       True,
    "dilated_conv": True,
    "upconv":       True
}
model_dir = './model_data/'
plot_dir = './conv_plots/'
pred_dir = './predictions/'
img_dir = './VOC2012-objects/JPEGImages/'
seg_dir = './VOC2012-objects/SegmentationObject/'

# Network parameters
layer_depth = 1
block_depth = 5  # minimally 2
num_hidden = 512
num_filters = 5
filter_size = [8, 8]
dropout_rate = 0.4

# Input parameters
img_width = 64
img_height = 64
max_img = 2000
batch_size = 1
train_percentage = 95

# Other parameters
steps = 500
step_log_interval = 50
num_epochs_train = 30
num_epochs_eval = 1
learning_rate = 0.001

# Hyperparameter optimizer parameters
hyperparameter1_search = {
    "Name": "layer_depth",  # choose from 'network parameters'
    "min_val": 1,
    "max_val": 4,
    "step": 1
}

hyperparameter2_search = {
    "Name": "block_depth",  # choose from 'network parameters'
    "min_val": 2,
    "max_val": 4,
    "step": 1
}

""" TODO:

Check output normalization

Include Gradient Noise

Include Early Stopping

"""
