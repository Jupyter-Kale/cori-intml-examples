"""
This module contains model and training code for the RPV classifier.
"""

# System
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Big data
import h5py

# Deep learning
import keras
from keras import layers, models


def load_file(filename, n_samples):
    with h5py.File(filename) as f:
        data_group = f['all_events']
        data = data_group['hist'][:n_samples][:,:,:,None]
        labels = data_group['y'][:n_samples]
        weights = data_group['weight'][:n_samples]
    return data, labels, weights

def build_model(input_shape,
                h1=64, h2=128, h3=256, h4=256, h5=512,
                optimizer=keras.optimizers.Adam, lr=0.001,
                use_horovod=True):
    # Define the NN layers
    inputs = layers.Input(shape=input_shape)
    conv_args = dict(kernel_size=(3, 3), activation='relu', padding='same')
    h = layers.Conv2D(h1, strides=1, **conv_args)(inputs)
    h = layers.Conv2D(h2, strides=2, **conv_args)(h)
    h = layers.Conv2D(h3, strides=1, **conv_args)(h)
    h = layers.Conv2D(h4, strides=2, **conv_args)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(h5, activation='relu')(h)
    outputs = layers.Dense(1, activation='sigmoid')(h)
    # Construct the distributed optimizer
    opt = optimizer(lr=lr)
    if use_horovod:
        import horovod.keras as hvd
        opt = hvd.DistributedOptimizer(opt)
    # Compile the model
    model = models.Model(inputs, outputs, 'RPVClassifier')
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_input, train_labels,
                valid_input, valid_labels,
                batch_size, n_epochs,
                lr_warmup_epochs=0, lr_reduce_patience=8,
                use_horovod=True):
    """Train the model"""
    callbacks = []
    if use_horovod:
        import horovod.keras as hvd
        callbacks += [
            # Horovod broadcast of initial variable states
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            # Average metrics among workers at the end of every epoch.
            hvd.callbacks.MetricAverageCallback(),
            # Scale learning rate down with factor 1/N and
            # increase to nominal after a specified number of epochs.
            # Comes from horovod examples and https://arxiv.org/abs/1706.02677
            hvd.callbacks.LearningRateWarmupCallback(
                warmup_epochs=lr_warmup_epochs, verbose=1),
        ]
    callbacks += [
        # Reduce the learning rate if training plateaues.
        keras.callbacks.ReduceLROnPlateau(
            patience=lr_reduce_patience, verbose=1),
    ]

    return model.fit(x=train_input, y=train_labels,
                     batch_size=batch_size, epochs=n_epochs,
                     validation_data=(valid_input, valid_labels),
                     callbacks=callbacks, verbose=2)
