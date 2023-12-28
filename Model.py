from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from pylab import *
from keras.callbacks import LearningRateScheduler
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import itertools
from sklearn.metrics import confusion_matrix
from keras import optimizers
from PIL import Image
import pandas as pd
import os

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return LearningRateScheduler(schedule)
    
vgg_conv = VGG19(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
model = models.Sequential()
    
for layer in vgg_conv.layers[:-5]:
    layer.trainable = False

    # Check the trainabl status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)


    model.add(vgg_conv)
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(2, activation='softmax'))

    optimizer = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])

    lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)