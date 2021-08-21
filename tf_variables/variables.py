#!/usr/bin/env python
""" Implementing low level Tensorflow variables """

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

v = tf.Variable(
        [-1, 2],
        dtype=tf.float32,
        name='test_vect')

v.assign([1101, 2202])

# print(v, '\n')
# print(v ** 2, '\n')
# print(v.numpy(), '\n')

# Tensors!
h = v + [9, 3]
print(h, '\n')

inputs = Input(shape=(5,))
h = Dense(16, activation='sigmoid')(inputs)
print(h, '\n')

outputs = Dense(10, activation='softmax', name='out_layer')(h)
print(outputs, '\n')

model = Model(inputs=inputs, outputs=outputs)
print(model.input)
print(model.output)

const = tf.constant([[5, 2], [1, 3]])
print('\n\nConstant tensor:', const, '\n')
print(const.numpy(), '\n')

print('\n', tf.ones(shape=(5, 4)), '\n')
print(tf.zeros(shape=(3, 8)), '\n')
