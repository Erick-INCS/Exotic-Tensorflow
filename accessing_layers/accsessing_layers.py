#!/usr/bin/env python
""" Accessing Tensorflow model layers"""

from tensorflow.keras.layers import (
        Dense, Input, Flatten, Conv1D,
        AveragePooling1D)
from tensorflow.keras.models import Model


inputs = Input(shape=(32, 1), name='input_layer')
h = Conv1D(3, 5, activation='relu', name='conv1d_layer')(inputs)
h = AveragePooling1D(3, name='avg_pool1d_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(20, activation='sigmoid', name='dense_layer')(h)

model = Model(inputs=inputs, outputs=outputs)

for ly_ix, layer in enumerate(model.layers):
    print('Layer', ly_ix + 1, ':\n', layer, '\n')

# Accessing weights
print(model.layers[1].weights, '\n')
print(model.layers[1].get_weights(), '\n')

# Kernel and bias
print('Kernel:', model.layers[1].kernel, '\n')
print('Bias:', model.layers[1].bias, '\n')

# Accessing by name
print('Accessing by name:', model.get_layer('conv1d_layer').bias, '\n')
