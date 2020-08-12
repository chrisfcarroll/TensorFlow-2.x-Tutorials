import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.data import Dataset
from tensorflow.keras.datasets import fashion_mnist
import os


def get_fashion_mnist_dataset() -> ((ndarray, ndarray), (ndarray, ndarray)):
    xtrain:ndarray;ytrain:ndarray;xval:ndarray;yval:ndarray

    def mnist_to_float32int32(x,y):
        return tf.cast(x,tf.float32)/255.0, tf.cast(y,tf.int32)

    (xtrain, ytrain,), (xval, yval) = fashion_mnist.load_data()
    print('(x,y) shapes: ', xtrain.shape, ytrain.shape)
    ytrain=tf.one_hot(ytrain, depth=10)
    yval=tf.one_hot(yval, depth=10)
    ds_train=tf.data.Dataset.from_tensor_slices((xtrain,ytrain))\
        .map(mnist_to_float32int32).shuffle(xtrain.shape[0]).batch(100)

    ds_val=tf.data.Dataset.from_tensor_slices((xval,yval))\
        .map(mnist_to_float32int32).shuffle(xval.shape[0]).batch(100)

    return ds_train,ds_val

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

train_ds, val_ds=get_fashion_mnist_dataset()

model=keras.Sequential([
    layers.Reshape(target_shape=(28*28,), input_shape=(28,28)),
    layers.Dense(200,activation='relu'),
    layers.Dense(200,activation='relu'),
    layers.Dense(200,activation='relu'),
    layers.Dense(10)
    ])

model.compile(
    optimizer=optimizers.Adam(),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(train_ds.repeat(), epochs=30, steps_per_epoch=500,
          validation_data=val_ds.repeat(),
          validation_steps=2)


keras.models.save_model(model,'mnistfashion4Dense_after30x500')

