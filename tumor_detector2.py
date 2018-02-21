import tensorflow as tf
import numpy as np
import parameters as par
import cv2
import random as rand
import time
import matplotlib.pyplot as plt
import scipy.signal as spsig
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')
# Display tensorflow info
tf.logging.set_verbosity(tf.logging.INFO)


class Tumor_detector:
    def __init__(self):
        self.start_time = time.time()

        # Weights for each layer
        self.w_dense = self.weight_variable(
            shape=[par.img_width * par.img_height * par.img_channels,
                   par.num_hidden],
            name="dense_layer_weights")

        self.w_output = self.weight_variable(
            shape=[par.num_hidden,
                   par.img_width * par.img_height * par.img_channels],
            name="output_layer_weights")

        # Biases for each layer
        self.b_dense = self.bias_variable(
            shape=[par.num_hidden],
            name="dense_layer_bias")

        self.b_output = self.bias_variable(
            shape=[par.img_width * par.img_height * par.img_channels],
            name="output_layer_bias")

        # Input layer
        self.input_layer = tf.placeholder(
            dtype=tf.float32,
            shape=[None, par.img_width, par.img_height, par.img_channels],
            name="input_layer")

        # Dense layer
        self.dense_layer = tf.nn.relu(
            features=tf.matmul(
                tf.reshape(
                    self.input_layer,
                    [-1, par.img_width * par.img_height * par.img_channels]),
                self.w_dense),
            name="dense_layer") + self.b_dense

        # Output layer
        self.output_layer = \
            tf.reshape(
                tensor=tf.matmul(self.dense_layer, self.w_output) + self.b_output,
                shape=[-1, par.img_width, par.img_height, par.img_channels],
                name="output_layer")

        print(self.output_layer)

        # Correct segmentations
        self.correct_output = tf.placeholder(
            dtype=tf.float32,
            shape=[None, par.img_width, par.img_height, par.img_channels],
            name="correct_output")

        print(self.output_layer)

        # Define loss and optimizer
        self.loss = tf.reduce_mean(
            tf.losses.absolute_difference(labels=self.correct_output,
                                          predictions=self.output_layer))

        # Define the training step
        optimizer = tf.train.GradientDescentOptimizer(par.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        # Define session and initialize variables
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Create a saver object
        self.saver = tf.train.Saver()


    def read_images(self):

        filenames = []
        seg_tensors = []
        img_tensors = []

        # Encode data
        seg_dir = './VOC2012-objects/SegmentationObject/'
        img_dir = './VOC2012-objects/JPEGImages/'
        for seg_file in os.listdir(seg_dir):
            filename = seg_file.split('.')[0]
            if len(filenames) > par.max_img:
                break
            if not os.path.isfile(img_dir + filename + '.jpg'):
                continue
            print("Now encoding image with filename {}".format(filename))
            filenames.append(filename)

            resized_seg = cv2.resize(src=cv2.imread(seg_dir + seg_file),
                                     dsize=(par.img_width, par.img_height))
            resized_img = cv2.resize(src=cv2.imread(img_dir + filename + '.jpg'),
                                     dsize=(par.img_width, par.img_height))

            resized_seg = resized_seg.reshape(par.img_width,
                                              par.img_height,
                                              par.img_channels)
            resized_img = resized_img.reshape(par.img_width,
                                              par.img_height,
                                              par.img_channels)

            resized_seg = np.asarray(resized_seg)
            resized_img = np.asarray(resized_img)

            resized_seg = resized_seg.astype(float) / 255
            resized_img = resized_img.astype(float) / 255

            seg_tensors.append(tf.convert_to_tensor(resized_seg, np.float32))
            img_tensors.append(tf.convert_to_tensor(resized_img, np.float32))

        total_img = len(filenames)
        n_train = par.train_percentage/100 * total_img

        self.train_img = []
        self.train_seg = []
        self.test_img = []
        self.test_seg = []

        for rand_idx in rand.sample(range(total_img), total_img):
            if len(self.train_img) < n_train:
                self.train_img.append(img_tensors[rand_idx])
                self.train_seg.append(seg_tensors[rand_idx])
            else:
                self.test_img.append(img_tensors[rand_idx])
                self.test_seg.append(seg_tensors[rand_idx])


    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)


    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def make_batches(self, imgs, segs):
        random_idx = rand.randrange(0, len(imgs)-par.batch_size)
        return imgs[random_idx: random_idx + par.batch_size], \
            segs[random_idx: random_idx + par.batch_size]


    def segment(self, feed, sess):
        output = self.output_layer.eval(session=sess,
                                        feed_dict={self.input_layer: feed})
        output = output[0]
        return output


    def train(self):

        for epoch in range(par.num_epochs_train):
            batch_img, batch_seg = self.make_batches(self.train_img,
                                                     self.train_seg)
            print("Batch_img: ")
            print(batch_img)
            print("self.input_layer: ")
            print(self.input_layer)
            print("self.correct_output: ")
            print(self.correct_output)

            self.sess.run(self.train_step,
                          feed_dict={
                                self.input_layer: batch_img,
                                self.correct_output: batch_seg})

            if epoch % 20 == 0:
                batch_img_eval, batch_seg_eval = self.make_batches(
                    self.test_img, self.test_seg)

                print(self.sess.run(self.accuracy,
                                    feed_dict={
                                        self.input_layer: batch_img_eval,
                                        self.correct_output: batch_seg_eval}))

        self.saver.save(self.sess, "tumor_detector_test")

        acc = 0
        for i in range(int(len(self.test_img)/self.batch_size)):

            batch_img_eval, batch_seg_eval = self.make_batches(
                self.test_img, self.test_seg)

            batch_acc = self.sess.run(self.accuracy,
                                      feed_dict={
                                          self.input_layer: batch_img_eval,
                                          self.correct_output: batch_seg_eval})
            acc += batch_acc
            if i % 10 == 0:
                print("Batch accuracy ({0}/{1}) = {2:.2f}\n".format(
                    int(len(self.test_img)/par.batch_size), batch_acc))

        acc = acc / (len(self.test_img) / par.batch_size)
        print("\n\nAccuracy = {0:.5f}\n".format(acc))
        end_time = time.time()
        print(time.strftime(
            "Time elapsed: \n%H hours, %M minutes, %S seconds\n",
            time.gmtime(int(end_time - self.start_time))))


if __name__ == '__main__':
    network = Tumor_detector()
    network.read_images()
    network.train()
    network.test()
