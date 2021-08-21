#!/usr/bin/env python
""" Accessing Tensorflow model layers """

from tensorflow.keras.layers import (
        Dense, Input, Flatten, Conv1D,
        AveragePooling1D)
from tensorflow.keras.models import Model, Sequential


inputs = Input(shape=(32, 1), name='input_layer')
h = Conv1D(3, 5, activation='relu', name='conv1d_layer')(inputs)
h = AveragePooling1D(3, name='avg_pool1d_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(1, activation='sigmoid', name='dense_layer')(h)

model = Model(inputs=inputs, outputs=outputs)

# New model without the last (Dense) layer
flatten_output = model.get_layer('flatten_layer').output
model2 = Model(inputs=model.input, outputs=flatten_output)

# New model with a new final dense layer
model3 = Sequential([
    model2,
    Dense(10, activation='softmax', name='new_dense_layer')])

#  With functional API
new_outputs = Dense(10, activation='softmax')(flatten_output)
#                                         or (model2.output)
model3 = Model(inputs=model.inputs, outputs=new_outputs)
