import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, metrics, regularizers

class VGG16(models.Model):

    def __init__(self,input_shape):
        """
        :param input_shape: [32,32,3]
        """
        super(VGG16,self).__init__()
        weight_decay= 0.000
        l2decay_regularizer = regularizers.l2(weight_decay)
        self.num_classes=10
        model=models.Sequential([layers.InputLayer(input_shape=input_shape)])
        model.add(layers.Conv2D(64, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(128, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(128, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(256, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(512, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(512, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3,3),
                                padding='same',
                                kernel_regularizer=l2decay_regularizer,
                                activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(512,
                               kernel_regularizer=l2decay_regularizer,
                               activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes))
        # model.add(layers.Activation('softmax'))

        self.model=model

    def call(self, x):
        x=self.model(x)
        return x
