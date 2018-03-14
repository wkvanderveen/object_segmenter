# bachpro

(NOTE: This is a work in progress. The network may not always work correctly. If you want a working version, it's best to pull the most recent version that has "Working Version" in the commit message.)

This is a residual segmentation network. It is used to detect objects in the VOC2012 dataset.

## Structure

This neural network has the following structure:
1) Input layer (a batch of 2-D images);

2) A variable number of "downward blocks", each consisting out of:
	a) repeated for a variable number of times:
          i) Same-size convolution layer
         ii) Batch normalization layer
        iii) Dense ReLU layer
         iv) Dropout layer
	b) a dilated convolution layer, to downscales the image size by a quarter of its size. The last downward block has no dilated convolution layer.

3) A variable number of "upward blocks", each consisting out of:
	a) an upconvolution layer, which upscales the image by to 4 times its size.
	b) alternating same-size convolution layers and ReLU dense layers, repeated for a variable number of times. The first of these convolution layers receives input from the previous upconvolutional layer, but also from the dilated convolution layer (or input layer) that scaled to the same image size.
    b) repeated for a variable number of times:
          i) Same-size convolution layer, receiving input from previous upconvolutional layer but also from the dilated convolution layer (or input layer) that scaled to the same image size.
         ii) Batch normalization layer
        iii) Dense ReLU layer
         iv) Dropout layer

4) An output layer using a sigmoidal activation function.


At the moment, I use the Adam optimizer and calculate the loss using a mean squared error calculation. This may change later on.

## Instructions

To use this network, please complete the following steps:

1) Ensure you have Python 3.6 installed on your system. The "csv", "pandas", "numpy", "tensorflow", "sklearn", and "cv2" modules are also required.

2) Download the VOC2012 dataset. (E.g., from "host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit").

3) Make an empty directory named "VOC2012-objects" in the same folder as where you downloaded the network files (i.e., object\_segmenter.py, parameters.py, hparam\_optimizer.py). In this directory, place the "JPEGImages" and "SegmentationObject" directories from the downloaded VOC2012 dataset.

3) You are now ready to run the network with "python3 object_segmenter.py".

4) Hyperparameter settings can be found (and modified) in parameters.py. If you experience problems during the previous step, changing the parameters to lower values might help, especially if you are running TensorFlow on a CPU.

5) In the parameter settings, you can enable various plot and visualization options. For instance, if you set "predict" to True, you can view the system output on a single (unknown) test image.

### Instructions on hyperparameter grid search 

The hparam_optimizer.py file can be used to find good hyperparameter settings for the network by doing a simple full 2-D grid search. It runs the network multiple times for various hyperparameter configurations.

To run this, change the relevant hyperparameters in "parameters.py" and then run "python3 hparam_optimizer.py".

### Using TensorBoard

To display information, such as the loss over time or a diagram of the network layers, you can you TensorBoard. 

1) Install it using "sudo -H pip3 install tensorboard".

2) Open a new terminal and navigate to the "bachpro" directory.

3) Enter "tensorboard --logdir ./model_data/"

4) Open a web browser and navigate to "http://localhost:6006".
