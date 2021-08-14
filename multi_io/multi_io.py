#!/usr/bin/env python
""" Implementing multiple imputs and outputs in Tensorflow and Keras """

import tensorflow as tf
from tensorflow.keras import Input, layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline

print('Using TF', tf.__version__)

pd_dat = pd.read_csv('/HDD/Data/CSV/diagnosis.csv')
dataset = pd_dat.values

# Build train and test split
X_train, X_test, Y_train, Y_test = train_test_split(
        dataset[:, :6],
        dataset[:, 6:],
        test_size=0.33)

# Assing corresponding IO
temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train =\
        np.transpose(X_train)

temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test =\
        np.transpose(X_test)

inflam_train, nephr_train = np.transpose(Y_train)
inflam_test, nephr_test = np.transpose(Y_test)


# Build the imput layes
shape_inputs = (1,)
temperature = Input(shape=shape_inputs, name='temp')
nausea_occurence = Input(shape=shape_inputs, name='nocc')
lumbar_pain = Input(shape=shape_inputs, name='lumbp')
urine_pushing = Input(shape=shape_inputs, name='up')
micturition_pains = Input(shape=shape_inputs, name='mict')
bis = Input(shape=shape_inputs, name='bis')

# Merge all input geateures into a single large vector
list_inputs = [
    temperature,
    nausea_occurence,
    lumbar_pain,
    urine_pushing,
    micturition_pains,
    bis
]
x = layers.concatenate(list_inputs)

# Logistic regression classifier for disease prediction
inflamation_pred = layers.Dense(1, activation='sigmoid', name='inflam')(x)
nephritis_pred = layers.Dense(1, activation='sigmoid', name='nephr')(x)

# Model outputs
list_outputs = [inflamation_pred, nephritis_pred]

model = tf.keras.Model(inputs=list_inputs, outputs=list_outputs)

# Plot the model
tf.keras.utils.plot_model(
        model,
        'multi_input_output_model.png',
        show_shapes=True)

# Compile the model
model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss={
            'inflam': 'binary_crossentropy',
            'nephr': 'binary_crossentropy'},
        metrics=['acc'],
        loss_weights=[1.0, 0.2])

# Fit model
inputs_train = {
        'temp': temp_train,
        'nocc': nocc_train,
        'lumbp': lumbp_train,
        'up': up_train,
        'mict': mict_train,
        'bis': bis_train}

outputs_train = {
        'inflam': inflam_train,
        'nephr': nephr_train}

history = model.fit(
        inputs_train,
        outputs_train,
        epochs=1_000,
        batch_size=128,
        verbose=False)

# Ploting learning curves
acc_keys = [k for k in history.history.keys() if k in ('inflam_acc', 'nephr_acc')]
loss_keys = [k for k in history.history.keys() if k not in acc_keys]

for k, v in history.history.items():
    if k in acc_keys:
        plt.figure(1)
        plt.plot(v)
    else:
        plt.figure(2)
        plt.plot(v)

plt.figure(1)
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(acc_keys, loc='upper right')
plt.savefig('Acc_vs_Epochs.jpg')

plt.figure(2)
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend('loss_keys', loc='upper right')
plt.savefig('Loss_vs_Epochs.jpg')

plt.show()

# Evaluate the model
print('\n\nResults:')
model.evaluate(
        [temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test],
        [inflam_test, nephr_test],
        verbose=2)
