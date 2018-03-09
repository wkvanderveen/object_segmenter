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
save_predictions = True  # 'True' in hyperparameter optimization
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
layer_depth = 2
block_depth = 3  # minimally 2
num_hidden = 128
num_filters = 10
filter_size = [10, 10]
dropout_rate = 0.4
optimizer = "Adam"

# Input parameters
img_width = 128
img_height = 128
max_img = 1000
batch_size = 5
train_percentage = 90

# Other parameters
steps = 10000
step_log_interval = 1000
num_epochs_train = 80
num_epochs_eval = 50
learning_rate = 0.001

# Hyperparameter optimizer parameters
hyperparameter1_search = {
    "Name": "num_hidden",  # choose from 'network parameters'
    "min_val": 20,
    "max_val": 60,
    "step": 40
}

hyperparameter2_search = {
    "Name": "num_filters",  # choose from 'network parameters'
    "min_val": 3,
    "max_val": 7,
    "step": 5
}



""" TODO:

Include dropout

Include batch normalization

Include Gradient Noise

Include Early Stopping



"""
