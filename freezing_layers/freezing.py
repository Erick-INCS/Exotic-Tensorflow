#!/usr/bin/env python3
""" Freezing layers to avoid unfix pretrained layers """

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
        Input, Dense, Flatten,
        Conv2D, MaxPooling2D)

# Froze specific layers
inputs = Input(shape=(8, 8, 1), name='input_layer')
h = Conv2D(
        16,
        3,
        activation='relu',
        name='conv2d_layer',
        trainable=False)(inputs)  # <- Here

h = MaxPooling2D(3, name='max_pooling2d_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(10, activation='softmax', name='softmax_layer')(h)

model = Model(inputs=inputs, outputs=outputs)

model.get_layer('conv2d_layer').trainable = False  # <- Here also

model.compile(loss='sparse_categorical_crossentropy')

# Froze the entire model
model = load_model('some_model.h5')
model.trainable = False

flatten_out = model.get_layer('flatten_layer').output
new_out = Dense(5, activation='softmax', name='new_out')(flatten_out)
new_model = Model(inputs=model.input, outputs=new_out)
new_model.compile(loss='sparse_categorical_crossentropy')
# sample
X_train, y_train = [], []
new_model.fit(X_train, y_train, epochs=10)
