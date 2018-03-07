# bachpro

(NOTE: This is a work in progress. The network may not always work correctly. If you want a working version, it's best to pull the most recent version that has "Working Version" in the commit message.)

This is a residual segmentation network. It is used to detect objects in the VOC2012 dataset.

## Structure

This neural network has the following structure:
1) Input layer (a batch of 2-D images);

2) A variable number of "downward blocks", each consisting out of:
	a) alternating same-size convolution layers and ReLU dense layers, repeated for a variable number of times;
	b) a dilated convolution layer, to downscales the image size by a quarter of its size. The last downward block has no dilated convolution layer.

3) A variable number of "upward blocks", each consisting out of:
	a) an upconvolution layer, which upscales the image by to 4 times its size.
	b) alternating same-size convolution layers and ReLU dense layers, repeated for a variable number of times. The first of these convolution layers receives input from the previous upconvolutional layer, but also from the dilated convolution layer (or input layer) that scaled to the same image size.

4) An output layer which is also a batch of 2-D images.


At the moment, I use the Adadelta optimizer and calculate the loss using a sums-of-squares method. These may (and probably will) change later on.

## Instructions

To use this network, please complete the following steps:

1) Ensure you have Python 3.6 installed on your system. The "csv", "pandas", "numpy", "tensorflow", and "cv2" modules are also required.

2) Download the VOC2012 dataset.

3) Make an empty directory named "VOC2012-objects" in the same folder as where you downloaded the network files (i.e., tumor\_detector.py, parameters.py, hparam\_optimizer.py). In this directory, place the "JPEGImages" and "SegmentationObject" directories from the downloaded VOC2012 dataset.

3) You are now ready to run the network with "python3 tumor_detector.py".

4) Hyperparameter settings can be found (and modified) in parameters.py. If you experience problems during the previous step, changing the parameters to lower values might help, especially if you are running TensorFlow on a CPU.

5) In the parameter settings, you can enable various plot and visualization options. For instance, if you set "predict" to True, you can view the system output on a single (unknown) test image.

### Instructions on random grid hyperparameter optimization

The hparam_optimizer.py file can be used to find good hyperparameter settings for the network by doing a simple full 2-D grid search. It runs the network multiple times for various hyperparameter configurations.

To run this, change the relevant parameters in "parameters.py" and then run "python3 hparam_optimizer.py".
