#!/usr/bin/env python
""" Customizing and existing model """

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# Imagenet!
# from tensorflow.keras.applications import VGG19
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import IPython.display as display
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load the model
vgg_model = load_model('/HDD/Data/models/VGG19.h5')

# Get the input layers and display the summary
vgg_input = vgg_model.input
vgg_layers = vgg_model.layers
vgg_model.summary()

# Build the model to access the layer outputs
layer_outputs = [layer.output for layer in vgg_layers]
features = Model(
        inputs=vgg_input,
        outputs=layer_outputs)

# Plot the model
tf.keras.utils.plot_model(
        features,
        'vgg19_model.png',
        show_shapes=True)

# Noise image
noise_img = np.random.random((1, 224, 224, 3)).astype('float32')
extracted_features = features(noise_img)

# Display the original image
img_path = 'cool_cat.jpg'
display.display(Image.open(img_path))

# Preoprocess the image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Extract the features
extracted_features = features(x)

# Visualize the input channels
fl = extracted_features[0]
print('\n fl.shape:', fl.shape)

imgs = fl[0, :, :]
plt.figure(figsize=(15, 15))
for n in range(3):
    ax = plt.subplot(1, 3, n + 1)
    plt.imshow(imgs[:, :, n])
    plt.axis('off')

plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig('cat.jpg')
