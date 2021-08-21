#!/usr/bin/env python3
""" Load dataste from Keras """

from tensorflow.keras.datasets import mnist
# from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# ~/.keras/datasets/mnist.npz

# (x_train, y_train), (x_test, y_test) = imdb.load_data(
#     num_words=100, max_len=100
# )
